import cv2
import numpy as np
import torch
import clip
import json
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config


class TrainingDataGenerator:
    """
    Generates (anchor_embedding, delta_vector, object_pool, target_embedding)
    tuples from any video. Self-supervised — no manual labels needed.
    Ground truth = CLIP(full_frame) which we already compute for validation.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
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
        for i, (pair, rd) in enumerate(delta.get("relation_deltas", {}).items()):
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
        config = Config()
        pipeline = ADVEPipeline(config)

        cap = cv2.VideoCapture(video_path)
        samples = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Let pipeline process the frame first to track coordinates & homography
            res = pipeline.process_frame(frame, frame_idx, no_validation=True)

            # Generate samples for delta frames (where we approximate instead of encode)
            if not res["is_anchor"] and pipeline.anchor_graph is not None:
                homography = pipeline._estimate_homography(pipeline.anchor_frame, frame)
                current_graph, delta = pipeline.delta_tracker.track(
                    frame, pipeline.anchor_graph, homography=homography
                )

                if len(delta.get("relation_deltas", {})) > 0:
                    gt_embedding = self.embed(frame)
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
                print(f"  Processed {frame_idx} frames, generated {len(samples)} samples")

        cap.release()

        # Ensure directory exists
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
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
        s = gen.generate_from_video(v, f"training/data/samples_{Path(v).stem}.json")
        all_samples.extend(s)

    os.makedirs("training/data", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_samples, f)
    print(f"Total: {len(all_samples)} samples saved to {args.output}")
