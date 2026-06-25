# ADVE — Product Master Document v2.0
## From Research to Best-in-Class Product
### Complete Technical Roadmap: Improvements · Architecture · Code · Deployment

---

# WHERE WE ARE

## Current ADVE (v1.0) — What Works

```
Core invention:    Spatial graph delta ΔG as proxy for semantic embedding change
Validated on:      Synthetic video (450 frames) + MOT17 (600 frames)
Key numbers:       96.67% savings (static), 60.33% savings (real video)
                   0.9923 cosine sim on MOT17, 65% VRAM reduction
Reconstruction:    Closed-form weighted average (no training required)
Dependencies:      CLIP ViT-B/32, YOLOv8n, ByteTrack
Status:            Research prototype, validated, GitHub live
```

## What v1.0 Cannot Do

```
1. Reconstruction is a weighted average — approximate, not learned
2. No camera motion compensation — panning camera confuses spatial delta
3. No streaming support — processes files, not live RTSP feeds
4. No search interface — produces embeddings, no way to query them
5. Not packaged — cannot pip install adve
6. Not edge-optimized — not tested on Jetson Nano or Raspberry Pi
7. Single camera — no multi-stream parallelism
8. No temporal memory — each anchor is independent, no rolling context
```

These are the gaps between research prototype and real product.
This document closes every one of them.

---

# PART 1: TECHNICAL IMPROVEMENTS
## v1.0 → v1.1 → v2.0 → v3.0

---

## v1.1 — Learned Reconstruction (Biggest Single Improvement)

### Why

Current reconstruction:
```
E_approx = weighted_average(object_embeddings, stability_weights)
Result: 0.9484 cosine sim on synthetic, 0.9923 on MOT17
```

The weighted average makes assumptions about how embeddings compose.
A small trained MLP learns the actual composition from data.
Expected result: 0.97+ cosine sim on both datasets.

### The Architecture

```
Input:
  E_anchor          (512-d CLIP embedding of anchor frame)
  ΔG_vector         (flattened spatial delta features)
  object_embs       (N × 512 object embeddings, pooled to 512)

ΔG_vector construction:
  For each object pair (i,j):
    [Δdistance_ij, Δangle_ij, Δsize_ratio_ij, magnitude_ij]
  Flatten + pad to fixed size 256-d

MLP Architecture:
  Input:  512 + 512 + 256 = 1280-d
  Layer 1: Linear(1280, 512) → LayerNorm → ReLU
  Layer 2: Linear(512, 256)  → LayerNorm → ReLU
  Layer 3: Linear(256, 512)  → L2 normalize
  Output: 512-d unit embedding

Loss: 1 - cosine_similarity(E_predicted, E_groundtruth)
      + 0.1 × MSE(E_predicted, E_groundtruth)
```

### Training Data Generation

```python
# training/generate_training_data.py

import cv2
import numpy as np
import torch
import clip
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

"""
Generates (anchor_embedding, delta_vector, object_pool, target_embedding)
tuples from any video. Self-supervised — no manual labels needed.
Ground truth = CLIP(full_frame) which we already compute for validation.
"""

class TrainingDataGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model, self.clip_prep = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        self.yolo = YOLO("yolov8n.pt")

    def embed(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            t = self.clip_prep(pil).unsqueeze(0).to(self.device)
            e = self.clip_model.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().flatten().astype(np.float32)

    def delta_to_vector(self, delta: dict, max_pairs: int = 32) -> np.ndarray:
        """Convert ΔG dict to fixed-size vector for MLP input."""
        vec = np.zeros(max_pairs * 4, dtype=np.float32)
        for i, (pair, rd) in enumerate(delta["relation_deltas"].items()):
            if i >= max_pairs:
                break
            base = i * 4
            vec[base]     = rd["delta_distance"]
            vec[base + 1] = rd["delta_angle"]
            vec[base + 2] = rd["delta_size_ratio"]
            vec[base + 3] = rd["magnitude"]
        return vec

    def pool_object_embeddings(
        self, objects: dict, max_objects: int = 8
    ) -> np.ndarray:
        """Pool per-object embeddings to fixed 512-d vector."""
        valid = [
            (obj.area, obj.embedding)
            for obj in objects.values()
            if obj.embedding is not None
        ]
        if not valid:
            return np.zeros(512, dtype=np.float32)

        valid.sort(key=lambda x: -x[0])  # sort by area, largest first
        valid = valid[:max_objects]

        weights = np.array([a for a, _ in valid], dtype=np.float32)
        weights /= weights.sum() + 1e-8
        embs = np.array([e for _, e in valid])

        return (weights[:, None] * embs).sum(axis=0).astype(np.float32)

    def generate_from_video(
        self, video_path: str, output_path: str,
        max_frames: int = 10000
    ):
        from adve_v2.core.pipeline import ADVEPipeline
        from adve_v2.core.config import Config

        config = Config()
        pipeline = ADVEPipeline(config)

        cap = cv2.VideoCapture(video_path)
        samples = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gt_embedding = self.embed(frame)

            # Only generate training samples for DELTA frames
            if (pipeline.anchor_graph is not None and
                    pipeline.frames_since_anchor > 0):

                current_graph, delta = pipeline.delta_tracker.track(
                    frame, pipeline.anchor_graph
                )

                if len(delta["relation_deltas"]) > 0:
                    sample = {
                        "anchor_emb":   pipeline.anchor_embedding.tolist(),
                        "delta_vec":    self.delta_to_vector(delta).tolist(),
                        "object_pool":  self.pool_object_embeddings(
                            current_graph.objects
                        ).tolist(),
                        "target_emb":   gt_embedding.tolist(),
                        "frame_idx":    frame_idx,
                    }
                    samples.append(sample)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Generated {len(samples)} samples from {frame_idx} frames")

        cap.release()

        with open(output_path, "w") as f:
            json.dump(samples, f)

        print(f"Saved {len(samples)} training samples → {output_path}")
        return samples


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--output", default="training/data/samples.json")
    args = p.parse_args()

    gen = TrainingDataGenerator()
    all_samples = []
    for v in args.videos:
        print(f"Processing: {v}")
        s = gen.generate_from_video(v, f"/tmp/samples_{Path(v).stem}.json")
        all_samples.extend(s)

    import json, os
    os.makedirs("training/data", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_samples, f)
    print(f"Total: {len(all_samples)} samples saved to {args.output}")
```

### The MLP Model

```python
# training/model.py

import torch
import torch.nn as nn


class ReconstructionMLP(nn.Module):
    """
    Learns to predict E(frame_t) from (E_anchor, ΔG_vector, object_pool).
    Replaces the closed-form weighted average in reconstructor.py.
    
    Once trained, drop this into reconstructor.py as a replacement.
    Expected improvement: 0.9484 → 0.97+ cosine sim.
    """

    def __init__(
        self,
        clip_dim:   int = 512,
        delta_dim:  int = 128,   # 32 pairs × 4 features
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = clip_dim + clip_dim + delta_dim  # anchor + pool + delta

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, clip_dim),
        )

        # Residual: start close to anchor, learn the delta
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        anchor_emb:  torch.Tensor,   # (B, 512)
        object_pool: torch.Tensor,   # (B, 512)
        delta_vec:   torch.Tensor,   # (B, 128)
    ) -> torch.Tensor:

        x = torch.cat([anchor_emb, object_pool, delta_vec], dim=-1)
        delta_pred = self.net(x)

        # Residual connection: output = anchor + learned_delta
        out = anchor_emb + self.residual_weight * delta_pred

        # Normalize to unit sphere (CLIP convention)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


class ReconstructionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine_loss = 1.0 - (pred * target).sum(dim=-1).mean()
        mse_loss    = ((pred - target) ** 2).sum(dim=-1).mean()
        return cosine_loss + 0.1 * mse_loss
```

### Training Script

```python
# training/train.py

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import ReconstructionMLP, ReconstructionLoss


class ReconstructionDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "anchor_emb":  torch.tensor(s["anchor_emb"],  dtype=torch.float32),
            "object_pool": torch.tensor(s["object_pool"], dtype=torch.float32),
            "delta_vec":   torch.tensor(s["delta_vec"][:128], dtype=torch.float32),
            "target_emb":  torch.tensor(s["target_emb"],  dtype=torch.float32),
        }


def train(json_path: str, epochs: int = 50, device: str = "cuda"):
    dataset    = ReconstructionDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    model     = ReconstructionMLP().to(device)
    criterion = ReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_sim  = 0.0

        for batch in dataloader:
            anchor  = batch["anchor_emb"].to(device)
            pool    = batch["object_pool"].to(device)
            delta   = batch["delta_vec"].to(device)
            target  = batch["target_emb"].to(device)

            pred = model(anchor, pool, delta)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sim = (pred * target).sum(dim=-1).mean().item()
            total_loss += loss.item()
            total_sim  += sim

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_sim  = total_sim  / len(dataloader)

        print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | cosine_sim={avg_sim:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "training/checkpoints/best_model.pt")
            print(f"  → Saved best model (sim={avg_sim:.4f})")

    print(f"\nTraining complete. Best cosine sim: {1 - best_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="training/data/samples.json")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    import os
    os.makedirs("training/checkpoints", exist_ok=True)
    train(args.data, args.epochs)
```

**Run sequence:**
```bash
# Step 1: Generate training data from multiple videos
python training/generate_training_data.py \
  --videos MOT17-02.mp4 MOT17-04.mp4 your_video.mp4 \
  --output training/data/samples.json

# Step 2: Train the MLP
python training/train.py --data training/data/samples.json --epochs 50

# Step 3: Validate improvement
python main.py --video MOT17-02-SDP-raw.webm --model training/checkpoints/best_model.pt
```

---

## v1.2 — Camera Motion Compensation

### Why

When the camera pans, every object "moves" in pixel space even if nothing changed in the scene. This false-triggers anchor refresh.

ADVE v1.0 on a panning camera: many false anchors → savings drop to 30-40%.
ADVE v1.2 with homography compensation: savings restored to 60-80%.

```python
# core/motion_compensation.py

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraMotionCompensator:
    """
    Estimates camera motion (homography) between consecutive frames.
    Normalizes object positions to remove camera motion before ΔG computation.
    
    Without this: camera pan → ADVE thinks all objects moved → false anchor refresh
    With this:    camera pan → positions normalized → ΔG stays small → correct behavior
    """

    def __init__(self, max_features: int = 500):
        self.orb       = cv2.ORB_create(max_features)
        self.matcher   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray]         = None  # last homography

    def estimate_homography(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Returns (H, is_camera_motion) where:
          H = 3x3 homography matrix (None if cannot estimate)
          is_camera_motion = True if motion is global (camera) not local (objects)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, False

        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            self.prev_gray = gray
            return None, False

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 8:
            self.prev_gray = gray
            return None, False

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        if H is None:
            self.prev_gray = gray
            return None, False

        # Determine if this is camera motion or object motion
        inlier_ratio = mask.sum() / len(mask)
        is_camera_motion = inlier_ratio > 0.7  # 70%+ inliers = global motion

        self.H = H
        self.prev_gray = gray
        return H, is_camera_motion

    def compensate_position(
        self,
        center: Tuple[float, float],
        H: np.ndarray
    ) -> Tuple[float, float]:
        """Transform a point by the inverse homography to remove camera motion."""
        H_inv   = np.linalg.inv(H)
        pt      = np.array([[[center[0], center[1]]]], dtype=np.float32)
        pt_comp = cv2.perspectiveTransform(pt, H_inv)
        return float(pt_comp[0, 0, 0]), float(pt_comp[0, 0, 1])

    def compensate_graph(self, graph, H: np.ndarray):
        """Apply compensation to all object centers in a SpatialGraph."""
        for obj in graph.objects.values():
            comp_center = self.compensate_position(obj.center, H)
            obj.center  = comp_center
        graph.build_relations()
        return graph
```

**How to integrate into pipeline.py:**

```python
# In ADVEPipeline.__init__:
from core.motion_compensation import CameraMotionCompensator
self.compensator = CameraMotionCompensator()

# In process_video, before compute_delta:
H, is_camera = self.compensator.estimate_homography(frame)
if H is not None and is_camera:
    current_graph = self.compensator.compensate_graph(current_graph, H)
# Now compute delta on compensated positions
delta = self.anchor_graph.compute_delta(current_graph)
```

---

## v1.3 — Streaming RTSP Support

Real product needs to process live camera feeds, not just video files.

```python
# core/stream.py

import cv2
import threading
import queue
import numpy as np
from typing import Optional, Callable


class RTSPStream:
    """
    Connects to a live RTSP camera feed.
    Feeds frames into ADVEPipeline continuously.
    Outputs embeddings via callback or WebSocket.
    
    Usage:
        stream = RTSPStream("rtsp://192.168.1.100:554/stream")
        stream.start(callback=lambda emb, ts: store_embedding(emb, ts))
    """

    def __init__(self, url: str, buffer_size: int = 30):
        self.url          = url
        self.buffer       = queue.Queue(maxsize=buffer_size)
        self._stop        = threading.Event()
        self._capture_thr = None
        self._process_thr = None
        self.pipeline     = None
        self.callback: Optional[Callable] = None

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise ConnectionError(f"Cannot connect to: {self.url}")

        print(f"Connected: {self.url}")
        frame_idx = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Stream interrupted, reconnecting...")
                cap.release()
                import time; time.sleep(2)
                cap = cv2.VideoCapture(self.url)
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                self.buffer.put_nowait((frame_idx, frame, timestamp))
            except queue.Full:
                pass  # drop frame if buffer full — live stream, don't lag

            frame_idx += 1

        cap.release()

    def _process_loop(self):
        from core.pipeline import ADVEPipeline
        from core.config   import Config

        pipeline = ADVEPipeline(Config())

        while not self._stop.is_set():
            try:
                frame_idx, frame, timestamp = self.buffer.get(timeout=1.0)
            except queue.Empty:
                continue

            result = pipeline.process_frame(frame, frame_idx)

            if self.callback and result:
                self.callback({
                    "frame_idx":     frame_idx,
                    "timestamp":     timestamp,
                    "embedding":     result["embedding"].tolist(),
                    "is_anchor":     result["is_anchor"],
                    "encoder_saved": not result["encoder_called"],
                })

    def start(self, callback: Optional[Callable] = None):
        self.callback     = callback
        self._capture_thr = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thr = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thr.start()
        self._process_thr.start()
        print(f"ADVE stream started: {self.url}")

    def stop(self):
        self._stop.set()
        self._capture_thr.join()
        self._process_thr.join()
        print("Stream stopped")


# Multi-camera manager
class MultiCameraManager:
    """
    Manages N simultaneous RTSP camera streams.
    Each runs its own ADVE pipeline independently.
    Embeddings from all cameras feed into a shared FAISS index.
    """

    def __init__(self, index_writer=None):
        self.streams     = {}
        self.index_writer = index_writer

    def add_camera(self, camera_id: str, rtsp_url: str):
        stream = RTSPStream(rtsp_url)
        stream.start(callback=lambda r: self._on_embedding(camera_id, r))
        self.streams[camera_id] = stream
        print(f"Camera added: {camera_id} → {rtsp_url}")

    def _on_embedding(self, camera_id: str, result: dict):
        if self.index_writer:
            self.index_writer.add(
                camera_id  = camera_id,
                frame_idx  = result["frame_idx"],
                timestamp  = result["timestamp"],
                embedding  = result["embedding"],
            )

    def remove_camera(self, camera_id: str):
        if camera_id in self.streams:
            self.streams[camera_id].stop()
            del self.streams[camera_id]

    def status(self) -> dict:
        return {
            cid: {"url": s.url, "buffer_size": s.buffer.qsize()}
            for cid, s in self.streams.items()
        }
```

---

## v2.0 — Vector Search Integration (Makes It a Product)

This is what turns ADVE from a research tool into something a customer pays for.
The customer does not care about cosine similarity. They care about:
"Find me every moment in my 1,000 hours of video where a person is near a door."

```python
# search/index.py

import numpy as np
import faiss
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    video_path:  str
    camera_id:   str
    timestamp:   float
    frame_idx:   int
    similarity:  float
    thumbnail:   Optional[str] = None  # base64 jpg


class ADVESearchIndex:
    """
    FAISS-powered semantic search over ADVE-generated embeddings.
    
    Two-layer storage:
      FAISS: fast approximate nearest-neighbor search on 512-d embeddings
      SQLite: metadata (video path, timestamp, camera ID) for each embedding
    
    Usage:
        index = ADVESearchIndex("my_index")
        index.add("camera_01", 1.5, 45, embedding_vector)
        results = index.search("person near door", k=10)
    """

    def __init__(self, index_dir: str, dim: int = 512):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.dim = dim

        # FAISS index — inner product on normalized vectors = cosine similarity
        faiss_path = self.index_dir / "embeddings.faiss"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)

        # SQLite for metadata
        self.db = sqlite3.connect(
            str(self.index_dir / "metadata.db"), check_same_thread=False
        )
        self._init_db()

        # Load CLIP for text queries
        self._clip_model  = None
        self._clip_prep   = None

    def _init_db(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                camera_id  TEXT,
                timestamp  REAL,
                frame_idx  INTEGER,
                is_anchor  INTEGER
            )
        """)
        self.db.commit()

    def add(
        self,
        video_path: str,
        camera_id:  str,
        timestamp:  float,
        frame_idx:  int,
        embedding:  np.ndarray,
        is_anchor:  bool = False,
    ):
        # Normalize to unit sphere
        emb = embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        self.faiss_index.add(emb.reshape(1, -1))

        self.db.execute(
            "INSERT INTO embeddings VALUES (NULL, ?, ?, ?, ?, ?)",
            (video_path, camera_id, timestamp, frame_idx, int(is_anchor))
        )
        self.db.commit()

    def add_batch(self, records: List[Dict]):
        """Batch insert for efficiency."""
        embeddings = np.array(
            [r["embedding"] for r in records], dtype=np.float32
        )
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms + 1e-8

        self.faiss_index.add(embeddings)

        self.db.executemany(
            "INSERT INTO embeddings VALUES (NULL, ?, ?, ?, ?, ?)",
            [(r["video_path"], r["camera_id"], r["timestamp"],
              r["frame_idx"], int(r.get("is_anchor", False)))
             for r in records]
        )
        self.db.commit()

    def search_by_text(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using a natural language query via CLIP text encoder."""
        import torch, clip
        if self._clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_prep = clip.load("ViT-B/32", device=device)

        device = next(self._clip_model.parameters()).device
        with torch.no_grad():
            tokens = clip.tokenize([query]).to(device)
            text_emb = self._clip_model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        query_vec = text_emb.cpu().numpy().astype(np.float32)
        return self._search(query_vec, k)

    def search_by_image(
        self, image: np.ndarray, k: int = 10
    ) -> List[SearchResult]:
        """Search using an image query — find similar scenes."""
        import torch, clip
        from PIL import Image

        if self._clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_prep = clip.load("ViT-B/32", device=device)

        device = next(self._clip_model.parameters()).device
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            t   = self._clip_prep(pil).unsqueeze(0).to(device)
            emb = self._clip_model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        query_vec = emb.cpu().numpy().astype(np.float32)
        return self._search(query_vec, k)

    def _search(self, query_vec: np.ndarray, k: int) -> List[SearchResult]:
        if self.faiss_index.ntotal == 0:
            return []

        k = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.db.execute(
                "SELECT video_path, camera_id, timestamp, frame_idx "
                "FROM embeddings WHERE id=?", (int(idx) + 1,)
            ).fetchone()

            if row:
                results.append(SearchResult(
                    video_path = row[0],
                    camera_id  = row[1],
                    timestamp  = row[2],
                    frame_idx  = row[3],
                    similarity = float(score),
                ))

        return results

    def save(self):
        faiss.write_index(
            self.faiss_index, str(self.index_dir / "embeddings.faiss")
        )

    def stats(self) -> dict:
        count = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        anchors = self.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE is_anchor=1"
        ).fetchone()[0]
        return {
            "total_embeddings": count,
            "anchor_frames":    anchors,
            "delta_frames":     count - anchors,
            "index_size_mb":    round(
                (self.index_dir / "embeddings.faiss").stat().st_size / 1e6, 2
            ) if (self.index_dir / "embeddings.faiss").exists() else 0,
        }
```

---

## v2.0 — FastAPI Product Server

This is the actual product that customers call.

```python
# api/server.py

"""
ADVE API Server
Run: uvicorn api.server:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /v1/index/video     — index a video file
  POST /v1/index/stream    — connect a live camera stream
  POST /v1/search/text     — search by text query
  POST /v1/search/image    — search by image
  GET  /v1/stats           — system statistics
  GET  /v1/health          — health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile, os, shutil, time

app = FastAPI(
    title="ADVE Video Intelligence API",
    description="Semantic video indexing and search using Anchor-Delta Video Embedding",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
from search.index import ADVESearchIndex
from core.pipeline import ADVEPipeline
from core.config   import Config
from core.stream   import MultiCameraManager

search_index = ADVESearchIndex("data/main_index")
camera_mgr   = MultiCameraManager(index_writer=search_index)


# ── Models ────────────────────────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query:     str
    k:         int = 10
    camera_id: Optional[str] = None

class StreamRequest(BaseModel):
    camera_id: str
    rtsp_url:  str

class SearchResult(BaseModel):
    video_path: str
    camera_id:  str
    timestamp:  float
    frame_idx:  int
    similarity: float


# ── Background indexing ───────────────────────────────────────────────────────

def index_video_task(video_path: str, video_id: str):
    """Runs in background after upload."""
    import cv2, numpy as np

    config   = Config()
    pipeline = ADVEPipeline(config)
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    batch    = []
    idx      = 0

    print(f"Indexing: {video_id}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result    = pipeline.process_frame(frame, idx)
        timestamp = idx / fps

        batch.append({
            "video_path": video_id,
            "camera_id":  video_id,
            "timestamp":  timestamp,
            "frame_idx":  idx,
            "embedding":  result["embedding"],
            "is_anchor":  result["is_anchor"],
        })

        if len(batch) >= 500:
            search_index.add_batch(batch)
            batch = []

        idx += 1

    if batch:
        search_index.add_batch(batch)

    cap.release()
    search_index.save()
    print(f"Indexed {idx} frames from {video_id}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/index/video")
async def index_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and index a video file. Returns immediately, indexes in background."""
    os.makedirs("data/uploads", exist_ok=True)
    dest = f"data/uploads/{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(index_video_task, dest, file.filename)

    return {
        "status":   "indexing_started",
        "video_id": file.filename,
        "message":  "Indexing running in background. Use /v1/stats to monitor."
    }


@app.post("/v1/index/stream")
async def add_stream(request: StreamRequest):
    """Connect a live RTSP camera stream."""
    try:
        camera_mgr.add_camera(request.camera_id, request.rtsp_url)
        return {"status": "connected", "camera_id": request.camera_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/search/text", response_model=List[SearchResult])
async def search_text(request: TextSearchRequest):
    """Search video content using a natural language query."""
    results = search_index.search_by_text(request.query, k=request.k)
    return [
        SearchResult(
            video_path = r.video_path,
            camera_id  = r.camera_id,
            timestamp  = r.timestamp,
            frame_idx  = r.frame_idx,
            similarity = r.similarity,
        )
        for r in results
    ]


@app.post("/v1/search/image", response_model=List[SearchResult])
async def search_image(file: UploadFile = File(...), k: int = 10):
    """Search for similar scenes using an image."""
    import cv2, numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = search_index.search_by_image(img, k=k)
    return [
        SearchResult(
            video_path = r.video_path,
            camera_id  = r.camera_id,
            timestamp  = r.timestamp,
            frame_idx  = r.frame_idx,
            similarity = r.similarity,
        )
        for r in results
    ]


@app.get("/v1/stats")
async def get_stats():
    return {
        "index":   search_index.stats(),
        "cameras": camera_mgr.status(),
    }


@app.get("/v1/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
```

---

## v2.0 — pip Package Structure

```
adve/
├── adve/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── anchor.py
│   │   ├── tracker.py
│   │   ├── reconstructor.py
│   │   ├── spatial_graph.py
│   │   └── motion_compensation.py
│   ├── search/
│   │   ├── __init__.py
│   │   └── index.py
│   ├── stream/
│   │   ├── __init__.py
│   │   └── rtsp.py
│   └── api/
│       ├── __init__.py
│       └── server.py
├── training/
│   ├── model.py
│   ├── train.py
│   └── generate_training_data.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_reconstruction.py
│   └── test_search.py
├── setup.py
├── pyproject.toml
└── README.md
```

```python
# setup.py

from setuptools import setup, find_packages

setup(
    name             = "adve",
    version          = "2.0.0",
    author           = "Asmitha",
    description      = "Anchor-Delta Video Embedding for efficient semantic video understanding",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/asmitha2025/ADVE",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "openai-clip",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "gpu":      ["faiss-gpu>=1.7.4"],
        "training": ["scikit-learn>=1.3.0"],
        "dev":      ["pytest", "black", "isort", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "adve=adve.core.main:main",
            "adve-api=adve.api.server:run",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)
```

---

## v3.0 — Edge Deployment (Jetson Nano)

```python
# edge/export_tensorrt.py

"""
Exports CLIP and YOLO to TensorRT FP16 for Jetson Nano.
Reduces inference latency by 4-8x vs PyTorch CPU.

Requirements: Jetson Nano with JetPack 5.x
Run ON the Jetson device, not on your desktop.
"""

import torch
import torch.onnx
import tensorrt as trt
from pathlib import Path


def export_clip_to_onnx(output_path: str = "edge/clip_vitb32.onnx"):
    """Export CLIP vision encoder to ONNX."""
    import clip

    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model.visual,
        dummy,
        output_path,
        input_names  = ["image"],
        output_names = ["embedding"],
        dynamic_axes = {"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version = 17,
    )
    print(f"Exported CLIP to ONNX: {output_path}")


def build_tensorrt_engine(
    onnx_path:   str,
    engine_path: str,
    fp16:        bool = True,
):
    """Build TensorRT engine from ONNX model."""
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"TensorRT engine saved: {engine_path}")


if __name__ == "__main__":
    Path("edge").mkdir(exist_ok=True)
    export_clip_to_onnx("edge/clip_vitb32.onnx")
    build_tensorrt_engine(
        "edge/clip_vitb32.onnx",
        "edge/clip_vitb32_fp16.engine",
        fp16=True,
    )
    print("Edge deployment ready.")
    print("Expected speedup: 4-6x vs PyTorch CPU on Jetson Nano")
    print("Expected VRAM:    ~180 MB (FP16 vs 330 MB FP32)")
```

---

# PART 2: DATASETS TO TEST NEXT

For paper credibility and product marketing, test on 3 more datasets:

```
Dataset 1: UCF-101 (Action Recognition)
  Why: Shows ADVE works on action videos, not just tracking
  Download: https://www.crcv.ucf.edu/data/UCF101.php
  What to measure: encoder savings + cosine sim across 101 action classes
  Expected: 70-85% savings (more motion than MOT17)

Dataset 2: BDD100K (Driving Video)
  Why: Autonomous vehicle use case, camera always moving
  Download: https://bdd-data.berkeley.edu
  What to measure: savings with motion compensation ON vs OFF
  Expected: motion compensation adds 15-25% savings on driving video

Dataset 3: VIRAT (Surveillance)
  Why: Direct proof for surveillance product pitch
  Download: https://viratdata.org
  What to measure: savings on mostly-static camera footage
  Expected: 90-95% savings (surveillance is mostly empty scenes)
```

---

# PART 3: BENCHMARKS TO ADD

```
Benchmark 1: Memory over time (not just peak)
  Plot VRAM usage over 1000 frames
  Show ADVE stays flat at 330 MB
  Show baseline climbs with longer videos
  This proves edge deployment stability

Benchmark 2: Latency per frame
  Anchor frame latency:  X ms (includes CLIP)
  Delta frame latency:   Y ms (YOLO only)
  Show Y << X
  This is the per-frame cost breakdown

Benchmark 3: Indexing throughput
  Hours of video indexed per hour of compute
  ADVE vs Full Embed vs Keyframe-30
  This is the number customers care about

Benchmark 4: Search quality (after indexing)
  10 text queries → retrieve top 10 results
  Human labels correct/incorrect
  Precision@10 metric
  Shows end-to-end product quality, not just embedding quality
```

---

# PART 4: PRODUCT ROADMAP

```
Month 1 — Library (pip install adve)
  ✓ Package all core modules
  ✓ Write docstrings and examples
  ✓ Publish to PyPI
  ✓ GitHub README with quickstart
  Goal: 100 GitHub stars, 50 pip installs

Month 2 — Learned Reconstruction (v1.1)
  ✓ Generate training data from MOT17 + UCF-101
  ✓ Train MLP reconstruction model
  ✓ Validate: cosine sim should hit 0.97+
  ✓ Release as adve==1.1.0
  Goal: paper update on arXiv with improved numbers

Month 3 — API Server (v2.0)
  ✓ FastAPI server packaged and deployable
  ✓ Docker image: docker pull asmitha2025/adve
  ✓ One-click deploy on AWS/GCP/DigitalOcean
  ✓ Simple dashboard for video upload + search
  Goal: first beta user (free tier)

Month 4 — First Customer
  ✓ Identify 3 surveillance/analytics companies in India
  ✓ Offer free 30-day pilot
  ✓ Measure their actual compute savings
  ✓ Collect testimonial
  Goal: first paid customer at ₹10K-25K/month

Month 5-6 — Edge Deployment (v3.0)
  ✓ Jetson Nano deployment tested
  ✓ TensorRT FP16 engine exported
  ✓ Power consumption measured
  ✓ Raspberry Pi 5 tested
  Goal: "runs on $50 hardware" claim is proven

Month 7-12 — Scale
  ✓ 5-10 paying customers
  ✓ Conference paper accepted
  ✓ Patent filed
  Goal: ₹5-25 lakh ARR, fundable thesis
```

---

# PART 5: WHAT TO BUILD FIRST

Given where you are right now, in this exact order:

```
This week (foundation):
  1. Package as pip library (setup.py + __init__.py)
  2. Generate training data from MOT17 video you have
  3. Train MLP reconstruction → see if cosine sim improves above 0.97

Next month (product):
  4. FastAPI server + Docker image
  5. Test on BDD100K or VIRAT dataset
  6. Build simple web dashboard (React or plain HTML)

Month 2-3 (customers):
  7. Find 1 surveillance company in Tamil Nadu or Bengaluru
  8. Offer free pilot
  9. Measure their real compute bill before/after

Month 3-6 (credibility):
  10. Jetson Nano deployment
  11. Conference submission with 3+ datasets
  12. Patent provisional filing
```

---

# PART 6: THE ONE NUMBER THAT SELLS IT

When you pitch to a customer, everything comes down to one number:

```
"We reduce your video AI compute bill by X%."

Fill in X with their actual workload.

Customer: "We have 500 CCTV cameras running 24/7"
Your calculation:
  500 cameras × 30fps × 86400 seconds = 1.296 billion frames/day
  Current cost at ₹3/1000 CLIP calls = ₹3.888 lakh/day
  With ADVE at 90% savings = ₹38,880/day
  Monthly savings = ₹1.06 Cr

You charge them: ₹5 lakh/month
They save:       ₹1.06 Cr/month
ROI for them:    21x return on subscription cost

That is the pitch. That is the product.
```

---

*ADVE Product Master Document v2.0*
*Next version: after MLP training results and first dataset expansion*
