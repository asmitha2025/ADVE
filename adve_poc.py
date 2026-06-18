"""
ADVE — Anchor Delta Video Embedding
Proof of Concept Validation

Hypothesis:
    E(frame_t) ≈ f(E_anchor_objects, ΔSpatialGraph)

    We can reconstruct scene-level semantics by tracking spatial
    relationships between objects — without re-running CLIP encoder
    on every frame.

Usage:
    python adve_poc.py <video_path> [--max-frames 300]
"""

import cv2
import torch
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine as cos_dist
from ultralytics import YOLO
import clip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ObjectNode:
    """Single tracked object with its semantic embedding."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    centroid: Tuple[float, float]
    embedding: np.ndarray             # CLIP embedding of this object's crop
    cls_name: str
    is_branch2: bool = False          # True = embedded during delta frame (new object)


class SceneGraph:
    """
    Spatial scene representation for one frame.

    Nodes  : tracked objects (ObjectNode)
    Edges  : pairwise normalized distance between every object pair

    Key insight: if edges don't change much, scene semantics don't change much.
    We quantify 'change' as max normalized distance-delta across all pairs.
    """

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.nodes: Dict[int, ObjectNode] = {}

    def add(self, node: ObjectNode):
        self.nodes[node.track_id] = node

    def _edges(self) -> Dict[Tuple[int, int], float]:
        """Pairwise normalized distances between all object pairs."""
        ids = sorted(self.nodes.keys())
        diag = np.sqrt(self.W ** 2 + self.H ** 2) + 1e-8
        edges = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.nodes[ids[i]]
                b = self.nodes[ids[j]]
                dx = b.centroid[0] - a.centroid[0]
                dy = b.centroid[1] - a.centroid[1]
                edges[(ids[i], ids[j])] = np.sqrt(dx * dx + dy * dy) / diag
        return edges

    def spatial_delta(self, other: "SceneGraph") -> float:
        """
        Max distance-change across shared object pairs (normalized 0-1).
        0.10 = objects moved 10% of frame diagonal relative to each other.
        """
        e_self = self._edges()
        e_other = other._edges()
        shared = set(e_self) & set(e_other)
        if not shared:
            return 0.0
        return max(abs(e_other[k] - e_self[k]) for k in shared)

    def reconstruct_embedding(self) -> Optional[np.ndarray]:
        """
        Core reconstruction:
            E_scene ≈ weighted_sum(object_embeddings)

        Weight = normalized_area × centrality_score
        Positions update every frame; embeddings come from anchor (no CLIP).
        """
        if not self.nodes:
            return None

        embs, weights = [], []
        for node in self.nodes.values():
            x1, y1, x2, y2 = node.bbox
            area = (x2 - x1) * (y2 - y1) / (self.W * self.H + 1e-8)
            cx = node.centroid[0] / self.W
            cy = node.centroid[1] / self.H
            centrality = max(0.1, 1.0 - np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2))
            embs.append(node.embedding)
            weights.append(area * centrality)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum() + 1e-8

        scene = np.zeros_like(embs[0])
        for w, e in zip(weights, embs):
            scene += w * e

        norm = np.linalg.norm(scene)
        return scene / norm if norm > 0 else scene


# ── Core ADVE System ───────────────────────────────────────────────────────────

class ADVE:
    """
    Anchor-Delta Video Embedding.

    KEYFRAME (anchor)
        Run YOLO + ByteTrack. Embed every crop with CLIP.
        Triggered on: frame 0, spatial_delta > threshold, frame budget exceeded.

    DELTA frame
        Track positions only (ByteTrack, no CLIP on known objects).
        Branch-2: run CLIP once for any NEW track ID entering scene.
        Reconstruct embedding from updated positions + cached anchor embeddings.
    """

    SPATIAL_THR    = 0.10   # Normalized dist-change that forces new keyframe
    MAX_DELTA_SPAN = 30     # Hard cap: force keyframe every N delta frames
    MIN_CONF       = 0.40   # YOLO confidence floor
    MIN_CROP_PX    = 16     # Skip crops smaller than 16x16 px

    def __init__(self, device: str = "cuda"):
        self.device = device

        print("Loading CLIP ViT-B/32 ...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        print("Loading YOLOv8n + ByteTrack ...")
        self.yolo = YOLO("yolov8n.pt")

        self.anchor_graph: Optional[SceneGraph] = None
        self.anchor_embs: Dict[int, np.ndarray] = {}
        self.delta_span: int = 0

        self.log: Dict = {
            "frame_ids": [], "similarities": [], "is_keyframe": [],
            "proc_ms": [], "frames": 0, "keyframes": 0, "branch2": 0,
        }

    def _embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = self.clip_preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            e = self.clip_model.encode_image(t).cpu().numpy().flatten().astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    def _track(self, frame: np.ndarray):
        return self.yolo.track(frame, persist=True, verbose=False, conf=self.MIN_CONF)[0]

    def _parse_box(self, box, W: int, H: int):
        if box.id is None:
            return None
        tid = int(box.id[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 - x1 < self.MIN_CROP_PX or y2 - y1 < self.MIN_CROP_PX:
            return None
        return tid, x1, y1, x2, y2, self.yolo.names[int(box.cls[0])]

    def _process_keyframe(self, frame: np.ndarray) -> SceneGraph:
        H, W = frame.shape[:2]
        graph = SceneGraph(W, H)
        res = self._track(frame)
        self.anchor_embs = {}

        for box in res.boxes:
            parsed = self._parse_box(box, W, H)
            if parsed is None:
                continue
            tid, x1, y1, x2, y2, cls_name = parsed
            emb = self._embed(frame[y1:y2, x1:x2])
            self.anchor_embs[tid] = emb
            graph.add(ObjectNode(
                track_id=tid, bbox=(x1, y1, x2, y2),
                centroid=((x1+x2)/2, (y1+y2)/2),
                embedding=emb, cls_name=cls_name,
            ))

        self.anchor_graph = graph
        self.delta_span = 0
        self.log["keyframes"] += 1
        return graph

    def _process_delta(self, frame: np.ndarray) -> Tuple[Optional[SceneGraph], bool]:
        H, W = frame.shape[:2]
        candidate = SceneGraph(W, H)
        res = self._track(frame)

        for box in res.boxes:
            parsed = self._parse_box(box, W, H)
            if parsed is None:
                continue
            tid, x1, y1, x2, y2, cls_name = parsed

            if tid in self.anchor_embs:
                emb = self.anchor_embs[tid]     # ← Zero CLIP cost
                is_b2 = False
            else:
                # Branch-2: new object, embed once then cache
                crop = frame[y1:y2, x1:x2]
                emb = self._embed(crop)
                self.anchor_embs[tid] = emb
                is_b2 = True
                self.log["branch2"] += 1

            candidate.add(ObjectNode(
                track_id=tid, bbox=(x1, y1, x2, y2),
                centroid=((x1+x2)/2, (y1+y2)/2),
                embedding=emb, cls_name=cls_name, is_branch2=is_b2,
            ))

        delta = self.anchor_graph.spatial_delta(candidate) if self.anchor_graph else 0.0
        force_kf = delta > self.SPATIAL_THR or self.delta_span >= self.MAX_DELTA_SPAN
        return (None, True) if force_kf else (candidate, False)

    def validate(self, video_path: str, max_frames: int = 300) -> Dict:
        """
        Main validation loop.
        Ground truth: full CLIP on every frame (validation only, not used in prod).
        ADVE output: anchor + delta reconstruction.
        Metric: cosine similarity between them.
        Target: mean >= 0.85 at >= 50% compute reduction.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        to_process = min(max_frames, total)

        print(f"\n  File    : {Path(video_path).name}")
        print(f"  FPS     : {fps:.1f}  |  Total: {total}  |  Processing: {to_process}\n")
        print(f"  {'Frame':>5}  {'Type':9}  {'Similarity':>10}  {'Δspan':>5}  {'ms':>6}")
        print("  " + "─" * 45)

        for idx in range(to_process):
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            self.log["frames"] += 1

            # Ground truth (validation only — in production this does not run)
            gt_emb = self._embed(frame)

            # ADVE path
            if idx == 0 or self.anchor_graph is None:
                graph = self._process_keyframe(frame)
                is_kf = True
            else:
                graph, needs_kf = self._process_delta(frame)
                if needs_kf:
                    graph = self._process_keyframe(frame)
                    is_kf = True
                else:
                    self.delta_span += 1
                    is_kf = False

            adve_emb = graph.reconstruct_embedding() if graph else None
            ms = (time.perf_counter() - t0) * 1000

            if adve_emb is not None:
                sim = float(1.0 - cos_dist(adve_emb, gt_emb))
                self.log["frame_ids"].append(idx)
                self.log["similarities"].append(sim)
                self.log["is_keyframe"].append(is_kf)
                self.log["proc_ms"].append(ms)

            if idx % 15 == 0 and self.log["similarities"]:
                s = self.log["similarities"][-1]
                tag = "KEYFRAME" if is_kf else "delta   "
                print(f"  {idx:5d}  {tag}  {s:10.4f}  {self.delta_span:5d}  {ms:6.0f}")

        cap.release()
        self._print_summary()
        return self.log

    def _print_summary(self):
        log = self.log
        sims = log["similarities"]
        total = log["frames"]
        kf = log["keyframes"]
        b2 = log["branch2"]
        clip_adve = kf + b2
        saved_pct = 100 * (1 - clip_adve / max(total, 1))
        mean_s = float(np.mean(sims)) if sims else 0
        min_s = float(np.min(sims)) if sims else 0
        above = sum(s >= 0.85 for s in sims)

        print("\n" + "═" * 54)
        print("  ADVE VALIDATION RESULTS")
        print("═" * 54)
        print(f"  Frames processed      : {total}")
        print(f"  Keyframes (anchors)   : {kf}")
        print(f"  Delta frames          : {total - kf}")
        print(f"  Branch-2 embeds       : {b2}  (new objects in delta frames)")
        print(f"  CLIP calls — standard : {total}")
        print(f"  CLIP calls — ADVE     : {clip_adve}")
        print(f"  Compute saved         : {saved_pct:.1f}%")
        print(f"  Mean cosine similarity: {mean_s:.4f}")
        print(f"  Min  cosine similarity: {min_s:.4f}")
        print(f"  Frames >= 0.85        : {above}/{len(sims)}")
        print("═" * 54)

        if mean_s >= 0.85 and saved_pct >= 50:
            print("  HYPOTHESIS VALIDATED")
            print(f"  {saved_pct:.0f}% compute saved at {mean_s:.3f} mean similarity")
        elif mean_s >= 0.75:
            print("  PARTIAL — good similarity, tune thresholds")
            print("  Try lowering SPATIAL_THR or MAX_DELTA_SPAN")
        else:
            print("  RECONSTRUCTION FAILS — gap in weighted-sum approach")
            print("  Next step: train a small delta-transform network")
        print("═" * 54)

    def plot(self, out: str = "adve_results.png"):
        log = self.log
        sims = log["similarities"]
        ids = log["frame_ids"]
        kfs = log["is_keyframe"]
        ms = log["proc_ms"]

        fig, axes = plt.subplots(3, 1, figsize=(14, 11), facecolor="#0d0d0d")
        fig.suptitle(
            "ADVE — Anchor Delta Video Embedding  |  Validation Results",
            fontsize=13, color="white", fontweight="bold", y=0.99
        )

        # Panel 1: Similarity timeline
        ax = axes[0]
        ax.set_facecolor("#111122")
        ax.plot(ids, sims, color="#00d4ff", lw=1.2, label="ADVE similarity")
        ax.axhline(0.85, color="#ff6b6b", ls="--", lw=1.2, label="0.85 threshold")
        ax.axhline(np.mean(sims), color="#69ff47", ls=":", lw=1.2,
                   label=f"mean = {np.mean(sims):.3f}")
        for f, k in zip(ids, kfs):
            if k:
                ax.axvline(f, color="orange", alpha=0.35, lw=0.8)
        h, _ = ax.get_legend_handles_labels()
        h.append(mpatches.Patch(color="orange", alpha=0.5, label="Keyframe reset"))
        ax.legend(handles=h, fontsize=9, facecolor="#111122", labelcolor="white")
        ax.set_ylabel("Cosine similarity", color="white")
        ax.set_title("Semantic Similarity: ADVE Reconstruction vs Full CLIP Ground Truth", color="white")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")

        # Panel 2: Compute savings
        ax = axes[1]
        ax.set_facecolor("#111122")
        adve_calls = log["keyframes"] + log["branch2"]
        saved = 100 * (1 - adve_calls / max(log["frames"], 1))
        bars = ax.bar(
            ["Standard\n(CLIP every frame)", "ADVE\n(Anchor + Delta)"],
            [log["frames"], adve_calls],
            color=["#ff6b6b", "#69ff47"], width=0.35, edgecolor="#0d0d0d"
        )
        ax.bar_label(bars, fmt="%d calls", padding=5, color="white", fontsize=11)
        ax.set_title(f"CLIP Encoder Calls  —  {saved:.1f}% Reduction", color="white")
        ax.set_ylabel("Full encoder calls", color="white")
        ax.grid(True, alpha=0.15, axis="y", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")

        # Panel 3: Latency distribution
        ax = axes[2]
        ax.set_facecolor("#111122")
        kf_ms = [t for t, k in zip(ms, kfs) if k]
        d_ms = [t for t, k in zip(ms, kfs) if not k]
        if kf_ms:
            ax.hist(kf_ms, bins=20, alpha=0.75, color="orange",
                    label=f"Keyframe  μ={np.mean(kf_ms):.0f} ms")
        if d_ms:
            ax.hist(d_ms, bins=20, alpha=0.75, color="#00d4ff",
                    label=f"Delta frame  μ={np.mean(d_ms):.0f} ms")
        ax.set_xlabel("Per-frame latency (ms)", color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title("Latency Distribution: Keyframe vs Delta Frame", color="white")
        ax.legend(facecolor="#111122", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"\nChart saved -> {out}")
        plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADVE Proof of Concept")
    ap.add_argument("video", help="Input video (MP4, AVI, MOV)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--spatial-thr", type=float, default=0.10,
                    help="Dist-delta threshold for keyframe trigger (default 0.10)")
    ap.add_argument("--max-delta-span", type=int, default=30,
                    help="Max delta frames before forced keyframe (default 30)")
    ap.add_argument("--output", default="adve_results.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*54}")
    print("  ADVE — Anchor Delta Video Embedding  PoC")
    print(f"  Device : {device.upper()}")
    print(f"{'='*54}")

    system = ADVE(device=device)
    system.SPATIAL_THR = args.spatial_thr
    system.MAX_DELTA_SPAN = args.max_delta_span

    system.validate(args.video, max_frames=args.max_frames)
    system.plot(args.output)
