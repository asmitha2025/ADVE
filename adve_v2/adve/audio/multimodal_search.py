"""
ADVE Multimodal Search
======================
Combines visual (CLIP image) and audio (Whisper + CLIP text) search results.
Returns ranked unified results with source labels.

This is what makes ADVE better than pure visual systems like Twelve Labs:
  - User searches "gradient descent"
  - Returns frames where it's SHOWN visually
  - AND frames where it's SAID in the audio
  - Merged, deduplicated, ranked
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MultimodalResult:
    video_path:  str
    timestamp:   float
    similarity:  float
    source:      str              # "visual", "audio", or "both"
    text:        str  = ""        # spoken text (audio results only)
    is_anchor:   bool = False     # was this an anchor frame (visual only)
    frame_idx:   int  = 0
    camera_id:   str  = ""
    objects:     List[str] = field(default_factory=list)


def merge_results(
    visual_results: list,
    audio_results:  list,
    min_gap:        float = 8.0,
    top_k:          int   = 5,
) -> List[MultimodalResult]:
    """
    Merge visual and audio search results into one ranked list.

    Strategy:
      1. Normalize scores within each modality (both to 0-1 range)
      2. Tag source: visual, audio, or both
      3. Merge timestamps within 3 seconds → "both" (strongest signal)
      4. Sort by score descending
      5. Temporal deduplicate: keep results ≥ min_gap seconds apart
      6. Return top_k

    "Both" results (visual AND audio match at same timestamp) get
    score boost because two independent signals agree.
    """

    # ── Calibrate scores absolute mapping ─────────────────────────────────
    def calibrate(results):
        if not results:
            return []
        for r in results:
            # Calibrate cosine similarity: [0.26, 0.32] mapped to [0.0, 1.0]
            r.similarity = max(0.0, min(1.0, (r.similarity - 0.26) / 0.06))
        return results

    visual_results = calibrate(list(visual_results))
    audio_results  = calibrate(list(audio_results))

    # ── Convert to MultimodalResult ───────────────────────────────────────
    merged = []

    for r in visual_results:
        merged.append(MultimodalResult(
            video_path = r.video_path,
            timestamp  = r.timestamp,
            similarity = r.similarity,
            source     = "visual",
            is_anchor  = getattr(r, "is_anchor", False),
            frame_idx  = getattr(r, "frame_idx", 0),
            camera_id  = getattr(r, "camera_id", ""),
            objects    = getattr(r, "objects", []),
        ))

    for r in audio_results:
        merged.append(MultimodalResult(
            video_path = r.video_path,
            timestamp  = r.timestamp,
            similarity = r.similarity,
            source     = "audio",
            text       = getattr(r, "text", ""),
            camera_id  = getattr(r, "camera_id", ""),
            objects    = [],
        ))

    # ── Merge close timestamps (visual + audio at same moment) ───────────
    # If visual and audio results are within 3 seconds → combine into "both"
    MERGE_GAP = 3.0
    to_remove = set()
    for i, r1 in enumerate(merged):
        for j, r2 in enumerate(merged):
            if i >= j or i in to_remove or j in to_remove:
                continue
            if r1.source == r2.source:
                continue
            if abs(r1.timestamp - r2.timestamp) <= MERGE_GAP:
                # Merge: take the earlier timestamp, boost score
                combined = MultimodalResult(
                    video_path = r1.video_path,
                    timestamp  = min(r1.timestamp, r2.timestamp),
                    similarity = min(1.0, (r1.similarity + r2.similarity) * 0.7),
                    source     = "both",
                    text       = r1.text or r2.text,
                    is_anchor  = r1.is_anchor or r2.is_anchor,
                    camera_id  = r1.camera_id,
                    objects    = r1.objects or r2.objects,
                )
                merged.append(combined)
                to_remove.add(i)
                to_remove.add(j)

    merged = [r for i, r in enumerate(merged) if i not in to_remove]

    # ── Sort by score ─────────────────────────────────────────────────────
    merged.sort(key=lambda r: r.similarity, reverse=True)

    # ── Temporal deduplicate ─────────────────────────────────────────────
    kept = []
    for result in merged:
        too_close = any(
            abs(result.timestamp - k.timestamp) < min_gap
            for k in kept
        )
        if not too_close:
            kept.append(result)
        if len(kept) >= top_k:
            break

    return kept


def format_source_badge(source: str) -> str:
    """HTML badge for result source."""
    badges = {
        "visual": '<span style="background:#1976D2;color:white;padding:2px 6px;border-radius:4px;font-size:11px">👁 Visual</span>',
        "audio":  '<span style="background:#7B1FA2;color:white;padding:2px 6px;border-radius:4px;font-size:11px">🎤 Audio</span>',
        "both":   '<span style="background:#2E7D32;color:white;padding:2px 6px;border-radius:4px;font-size:11px">⚡ Visual + Audio</span>',
    }
    return badges.get(source, "")
