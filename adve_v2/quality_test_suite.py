"""
ADVE Production Quality Test Suite
=====================================
Tests ADVE against 10 diverse video types to find where quality breaks.

This is the foundation of a production-ready product.
You cannot sell something you have not tested comprehensively.

Run: python quality_test_suite.py

Outputs:
  quality_report.json   — per-type metrics
  quality_report.md     — human-readable report
  quality_plots/        — per-type plots
"""

import cv2
import json
import time
import numpy as np
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from pathlib import Path

# Add the directory containing this script to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class VideoTypeResult:
    video_type:        str
    video_path:        str
    total_frames:      int
    processed_frames:  int
    anchor_frames:     int
    encoder_savings:   float
    mean_cosine_sim:   float
    min_cosine_sim:    float
    pct_above_085:     float
    indexing_time_sec: float
    search_results:    List[dict] = field(default_factory=list)
    issues_found:      List[str]  = field(default_factory=list)
    verdict:           str        = ""
    similarity_history: List[float] = field(default_factory=list)


class QualityTestSuite:

    VIDEO_TYPES = {
        "lecture":      "Static camera, talking head, whiteboard content",
        "sports":       "Fast motion, multiple players, rapid cuts",
        "news":         "Anchor person, lower thirds, graphics overlay",
        "meeting":      "Multiple faces, static background, presentation",
        "tutorial":     "Hands close-up, objects, demonstration",
        "documentary":  "Varied scenes, narration, slow transitions",
        "product_demo": "Close-up objects, product focus, indoor",
        "short_film":   "Dramatic cuts, varied lighting, narrative",
        "cctv":         "Static camera, sparse motion, wide angle",
        "street":       "Pedestrians, vehicles, outdoor, varied motion",
    }

    SEARCH_QUERIES = {
        "lecture":      ["person at whiteboard", "mathematical formula", "student question"],
        "sports":       ["player running", "ball in play", "goal or score"],
        "news":         ["news anchor speaking", "graphic on screen", "interview"],
        "meeting":      ["person presenting", "slide on screen", "group discussion"],
        "tutorial":     ["hands demonstrating", "close up object", "step by step"],
        "documentary":  ["landscape scene", "person speaking", "wildlife or nature"],
        "product_demo": ["product close up", "feature demonstration", "comparison"],
        "short_film":   ["dramatic scene", "character speaking", "action sequence"],
        "cctv":         ["person entering", "vehicle moving", "empty scene"],
        "street":       ["pedestrian walking", "car passing", "crowd"],
    }

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_device = "cpu"
        self._clip_model  = None
        self._clip_prep   = None
        self.results: List[VideoTypeResult] = []

    def _load_models(self):
        if self._clip_model is None:
            import clip
            self._clip_model, self._clip_prep = clip.load("ViT-B/32", device=self._clip_device)
            self._clip_model.eval()

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            t   = self._clip_prep(pil).unsqueeze(0).to(self._clip_device)
            e   = self._clip_model.encode_image(t)
            e   = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().flatten().astype(np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        import clip, torch
        with torch.no_grad():
            tokens = clip.tokenize([text[:77]], truncate=True).to(self._clip_device)
            e      = self._clip_model.encode_text(tokens)
            e      = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().flatten().astype(np.float32)

    def test_video(
        self,
        video_path:  str,
        video_type:  str,
        max_frames:  int = 600,
    ) -> VideoTypeResult:

        self._load_models()

        # Initialize ADVEPipeline
        from adve.core.config import Config
        from adve.core.pipeline import ADVEPipeline
        
        config = Config()
        config.DEVICE = self.device
        config.CLIP_DEVICE = "cpu"
        config.YOLO_DEVICE = self.device
        
        # Resolve MLP path relative to quality_test_suite.py
        adve_v2_dir = os.path.dirname(os.path.abspath(__file__))
        config.MLP_MODEL_PATH = os.path.join(adve_v2_dir, "training", "checkpoints", "best_model.pt")
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        adve_pipeline = ADVEPipeline(
            config,
            clip_model=self._clip_model,
            clip_preprocess=self._clip_prep
        )

        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip  = max(1, int(fps // 5))

        embeddings  = []
        gt_embs     = []
        frame_idx   = 0
        processed   = 0
        anchors     = 0
        issues      = []

        t_start = time.time()

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process EVERY frame through the actual pipeline
            result = adve_pipeline.process_frame(frame, frame_idx, no_validation=True)

            if frame_idx % skip == 0:
                gt_emb = self._embed(frame)
                gt_embs.append(gt_emb)
                embeddings.append(result["embedding"])
                processed += 1

            if result["is_anchor"]:
                anchors += 1

            if frame_idx % 50 == 0:
                print(f"    [{video_type}] Processed {frame_idx}/{max_frames} frames...")
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t_start

        # Compute cosine similarities
        sims = []
        for approx, gt in zip(embeddings, gt_embs):
            if not (approx is not None and gt is not None):
                continue
            sim = float(np.dot(approx, gt) / (
                np.linalg.norm(approx) * np.linalg.norm(gt) + 1e-8
            ))
            sims.append(sim)

        delta_sims = sims[1:]  # skip first anchor (trivially 1.0)
        savings    = round((1 - anchors / max(frame_idx, 1)) * 100, 1)

        mean_sim   = float(np.mean(delta_sims))  if delta_sims else 1.0
        min_sim    = float(np.min(delta_sims))   if delta_sims else 1.0
        pct_above  = (sum(1 for s in delta_sims if s >= 0.85) / len(delta_sims) * 100) if delta_sims else 100.0

        # Issue detection
        if mean_sim < 0.85:
            issues.append(f"LOW ACCURACY: mean cosine sim {mean_sim:.3f} below 0.85 threshold")
        if min_sim < 0.70:
            issues.append(f"QUALITY FLOOR: min cosine sim {min_sim:.3f} — some frames very poorly reconstructed")
        if savings < 30:
            issues.append(f"LOW SAVINGS: only {savings}% — video has high scene change rate")
        if anchors == processed:
            issues.append("NO DELTA FRAMES: anchor refresh triggered every frame — threshold needs tuning")

        # Verdict
        if mean_sim >= 0.90 and min_sim >= 0.80:
            verdict = "EXCELLENT"
        elif mean_sim >= 0.85 and min_sim >= 0.75:
            verdict = "GOOD"
        elif mean_sim >= 0.80:
            verdict = "ACCEPTABLE"
        else:
            verdict = "NEEDS WORK"

        # Test search
        search_results = []
        if embeddings:
            import faiss
            idx = faiss.IndexFlatIP(512)
            idx.add(np.vstack(embeddings).astype(np.float32))

            queries = self.SEARCH_QUERIES.get(video_type, ["person", "object"])
            for q in queries[:3]:
                q_emb = self._embed_text(q).reshape(1, -1)
                k = min(3, idx.ntotal)
                scores, _ = idx.search(q_emb, k)
                top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0
                search_results.append({
                    "query":     q,
                    "top_score": round(top_score, 4),
                    "works":     top_score > 0.20,
                })

        # Free GPU VRAM before returning to prevent OOM across runs
        del adve_pipeline
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return VideoTypeResult(
            video_type        = video_type,
            video_path        = video_path,
            total_frames      = frame_idx,
            processed_frames  = processed,
            anchor_frames     = anchors,
            encoder_savings   = savings,
            mean_cosine_sim   = round(mean_sim, 4),
            min_cosine_sim    = round(min_sim, 4),
            pct_above_085     = round(pct_above, 1),
            indexing_time_sec = round(elapsed, 1),
            search_results    = search_results,
            issues_found      = issues,
            verdict           = verdict,
            similarity_history = delta_sims,
        )

    def run_all(self, video_map: Dict[str, str]) -> str:
        print(f"\n{'='*60}")
        print(f"  ADVE Quality Test Suite (Actual Pipeline)")
        print(f"  Testing {len(video_map)} video types")
        print(f"{'='*60}\n")

        for vtype, vpath in video_map.items():
            if not os.path.exists(vpath):
                print(f"  ⚠️  Skipping {vtype}: file not found ({vpath})")
                continue

            print(f"  Testing: {vtype} ({Path(vpath).name})...")
            result = self.test_video(vpath, vtype)
            self.results.append(result)

            verdict_icon = {
                "EXCELLENT": "✅",
                "GOOD":      "✅",
                "ACCEPTABLE": "⚠️",
                "NEEDS WORK": "❌",
            }.get(result.verdict, "?")

            print(
                f"  {verdict_icon} {vtype:<15} "
                f"sim={result.mean_cosine_sim:.4f} "
                f"savings={result.encoder_savings}% "
                f"verdict={result.verdict}"
            )
            if result.issues_found:
                for issue in result.issues_found:
                    print(f"     ⚠️  {issue}")

        return self._generate_report()

    def _generate_report(self) -> str:
        os.makedirs("quality_plots", exist_ok=True)

        self._generate_plots()

        # JSON report
        report_results = []
        for r in self.results:
            d = asdict(r)
            if "similarity_history" in d:
                del d["similarity_history"]
            report_results.append(d)

        report_data = {
            "summary": self._summary(),
            "results": report_results,
        }
        with open("quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Markdown report
        md = self._markdown_report()
        with open("quality_report.md", "w", encoding="utf-8") as f:
            f.write(md)

        print(f"\n{'='*60}")
        print(f"  Reports saved:")
        print(f"    quality_report.json")
        print(f"    quality_report.md")
        print(f"    quality_plots/ (plots generated)")
        print(f"{'='*60}\n")

        return md

    def _generate_plots(self):
        try:
            import matplotlib.pyplot as plt
            for r in self.results:
                if not hasattr(r, "similarity_history") or not r.similarity_history:
                    continue
                plt.figure(figsize=(10, 4))
                plt.plot(r.similarity_history, label="Cosine Similarity", color="#1f77b4", linewidth=2)
                plt.axhline(y=0.85, color="r", linestyle="--", alpha=0.7, label="Threshold (0.85)")
                
                plt.title(f"Reconstruction Quality: {r.video_type} ({r.verdict})", fontsize=14, fontweight="bold", pad=15)
                plt.xlabel("Frame Index (sampled)", fontsize=11, labelpad=8)
                plt.ylabel("Cosine Similarity", fontsize=11, labelpad=8)
                plt.ylim(0.0, 1.05)
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
                
                plt.tight_layout()
                plot_path = os.path.join("quality_plots", f"{r.video_type}_similarity.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
        except Exception as e:
            print(f"  ⚠️  Could not generate plots: {e}")

    def _summary(self) -> dict:
        if not self.results:
            return {}
        mean_sims = [r.mean_cosine_sim for r in self.results]
        savings   = [r.encoder_savings for r in self.results]
        verdicts  = [r.verdict for r in self.results]
        return {
            "total_tested":       len(self.results),
            "excellent_or_good":  sum(1 for v in verdicts if v in ("EXCELLENT","GOOD")),
            "needs_work":         sum(1 for v in verdicts if v == "NEEDS WORK"),
            "mean_cosine_sim":    round(float(np.mean(mean_sims)), 4),
            "mean_savings":       round(float(np.mean(savings)), 1),
            "production_ready":   all(r.mean_cosine_sim >= 0.85 for r in self.results),
        }

    def _markdown_report(self) -> str:
        s   = self._summary()
        lines = [
            "# ADVE Quality Test Report",
            "",
            "## Summary",
            f"- Videos tested: {s.get('total_tested', 0)}",
            f"- Excellent/Good: {s.get('excellent_or_good', 0)}",
            f"- Needs Work: {s.get('needs_work', 0)}",
            f"- Mean cosine sim: {s.get('mean_cosine_sim', 0)}",
            f"- Mean encoder savings: {s.get('mean_savings', 0)}%",
            f"- Production ready: {'✅ YES' if s.get('production_ready') else '❌ NO'}",
            "",
            "## Per-Type Results",
            "",
            "| Video Type | Mean Sim | Min Sim | Savings | Search | Verdict |",
            "|------------|----------|---------|---------|--------|---------|",
        ]

        for r in self.results:
            search_ok = all(q["works"] for q in r.search_results)
            lines.append(
                f"| {r.video_type:<15} | {r.mean_cosine_sim:.4f} | "
                f"{r.min_cosine_sim:.4f} | {r.encoder_savings}% | "
                f"{'✅' if search_ok else '❌'} | {r.verdict} |"
            )

        lines += [
            "",
            "## Issues Found",
            "",
        ]

        for r in self.results:
            if r.issues_found:
                lines.append(f"### {r.video_type}")
                for issue in r.issues_found:
                    lines.append(f"- {issue}")
                lines.append("")

        lines += [
            "",
            "## Quality Plots",
            "",
        ]
        for r in self.results:
            plot_file = f"quality_plots/{r.video_type}_similarity.png"
            if os.path.exists(plot_file):
                lines.append(f"### {r.video_type.capitalize()} Quality Plot")
                lines.append(f"![{r.video_type} plot]({plot_file})")
                lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+",
                   help="video_type:path pairs e.g. lecture:videos/lec.mp4 cctv:cam.webm")
    p.add_argument("--quick",  action="store_true",
                   help="Test with existing MOT17 video only")
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    suite = QualityTestSuite(device=dev)

    if args.quick:
        video_map = {}
        for vtype, vpath in [
            ("cctv",    "Input video/MOT17-02-SDP-raw.webm"),
            ("street",  "Input video/MOT17-02-SDP-raw.webm"),
        ]:
            if os.path.exists(vpath):
                video_map[vtype] = vpath

        if not video_map:
            print("No videos found. Provide --videos argument.")
        else:
            suite.run_all(video_map)

    elif args.videos:
        video_map = {}
        for pair in args.videos:
            parts = pair.split(":", 1)
            if len(parts) == 2:
                video_map[parts[0]] = parts[1]
        suite.run_all(video_map)

    else:
        print("Usage:")
        print("  Quick test:  python quality_test_suite.py --quick")
        print("  Full test:   python quality_test_suite.py --videos lecture:lec.mp4 sports:match.mp4")
