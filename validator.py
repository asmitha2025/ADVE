import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from config import Config


class Validator:
    """
    Records per-frame cosine similarity between reconstructed and ground-truth
    embeddings, computes summary statistics, and produces plots.
    """

    def __init__(self, config: Config):
        self.config  = config
        self.records: List[Dict] = []

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def log(
        self,
        frame_idx:       int,
        reconstructed:   np.ndarray,
        ground_truth:    Optional[np.ndarray],
        is_anchor:       bool,
        delta_magnitude: float,
        encoder_called:  bool,
    ) -> float:
        if ground_truth is None:
            sim = 1.0
        else:
            sim = float(
                np.dot(reconstructed, ground_truth) /
                (np.linalg.norm(reconstructed) * np.linalg.norm(ground_truth) + 1e-8)
            )

        self.records.append({
            "frame":           frame_idx,
            "cosine_sim":      sim,
            "is_anchor":       is_anchor,
            "delta_magnitude": delta_magnitude,
            "encoder_called":  encoder_called,
        })

        return sim

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> Dict:
        delta_sims = [r["cosine_sim"] for r in self.records if not r["is_anchor"]]
        all_sims   = [r["cosine_sim"] for r in self.records]

        encoder_calls = sum(1 for r in self.records if r["encoder_called"])
        total_frames  = len(self.records)

        above = sum(1 for s in delta_sims if s >= self.config.SUCCESS_THRESHOLD)

        summary = {
            "total_frames":          total_frames,
            "encoder_calls":         encoder_calls,
            "delta_frames":          total_frames - encoder_calls,
            "encoder_savings_pct":   round((1 - encoder_calls / total_frames) * 100, 2),
            "mean_cosine_sim":       round(float(np.mean(all_sims)), 4)   if all_sims   else 0,
            "mean_delta_cosine_sim": round(float(np.mean(delta_sims)), 4) if delta_sims else 0,
            "min_delta_cosine_sim":  round(float(np.min(delta_sims)), 4)  if delta_sims else 0,
            "pct_above_threshold":   round(above / len(delta_sims) * 100, 2) if delta_sims else 0,
            "hypothesis_validated":  (
                float(np.mean(delta_sims)) >= self.config.SUCCESS_THRESHOLD
                if delta_sims else False
            ),
        }
        return summary

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot(self, out_path: str) -> None:
        frames   = [r["frame"]      for r in self.records]
        sims     = [r["cosine_sim"] for r in self.records]
        anchors  = [r["frame"]      for r in self.records if r["is_anchor"]]
        enc_mask = [1 if r["encoder_called"] else 0 for r in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle("ADVE — Anchor-Delta Video Embedding | Validation", fontsize=13)

        # --- Plot 1: Cosine similarity ---
        axes[0].plot(frames, sims, color="#2196F3", linewidth=1.2, label="Cosine Similarity")
        axes[0].axhline(
            self.config.SUCCESS_THRESHOLD, color="green",
            linestyle="--", linewidth=1, label=f"Threshold ({self.config.SUCCESS_THRESHOLD})"
        )
        for a in anchors:
            axes[0].axvline(a, color="red", alpha=0.25, linewidth=0.8)
        axes[0].set_title("Reconstructed vs Ground-Truth Embedding Similarity")
        axes[0].set_ylabel("Cosine Similarity")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.25)

        # --- Plot 2: Delta magnitude ---
        delta_mags = [r["delta_magnitude"] for r in self.records]
        axes[1].plot(frames, delta_mags, color="#FF5722", linewidth=1, label="ΔG Magnitude")
        axes[1].axhline(
            self.config.SPATIAL_THRESHOLD, color="orange",
            linestyle="--", linewidth=1, label="Anchor Trigger Threshold"
        )
        axes[1].set_title("Spatial Graph Delta Magnitude per Frame")
        axes[1].set_ylabel("ΔG Magnitude")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)

        # --- Plot 3: Encoder calls ---
        axes[2].fill_between(frames, enc_mask, alpha=0.6, color="#E91E63", label="CLIP Called")
        axes[2].fill_between(
            frames,
            [1 - m for m in enc_mask],
            alpha=0.3, color="#4CAF50", label="CLIP Skipped (Saved)"
        )
        axes[2].set_title("CLIP Encoder Calls (Red = Called, Green = Saved)")
        axes[2].set_xlabel("Frame Index")
        axes[2].set_ylabel("Encoder Called")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_json(self, out_path: str) -> None:
        with open(out_path, "w") as f:
            json.dump(
                {"summary": self.summarize(), "frames": self.records},
                f, indent=2
            )
