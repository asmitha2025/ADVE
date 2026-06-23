"""
ADVE Unified Search Engine
============================
Combines three independent search signals:

  Signal 1: CLIP Visual    — what the scene LOOKS LIKE
  Signal 2: OCR Text       — what text is READABLE in the frame
  Signal 3: Whisper Audio  — what is SPOKEN at that moment

Each signal finds different things. Merging them finds everything.

Example: user searches "backpropagation gradient formula"

  CLIP visual:  finds frames visually similar to "math diagrams"
                might find the right slide if it has enough visual context

  OCR text:     finds frames containing the word "backpropagation"
                or "∂L/∂w" or "gradient" directly on screen
                exact match — never misses it if text is visible

  Whisper audio: finds moments where someone SAYS "backpropagation"
                 or "gradient descent" or "the formula for..."

Merged: returns all three sets, deduplicated, ranked
        "both" signals agreeing → highest confidence result

This is what makes ADVE better than every competitor:
  Twelve Labs:  CLIP visual only
  Gemini video: CLIP visual + some audio, expensive
  ADVE:         CLIP visual + OCR text + Whisper audio, 5-10x cheaper
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UnifiedResult:
    video_id:    str
    timestamp:   float
    similarity:  float
    sources:     List[str]       # which signals matched: ["visual", "ocr", "audio"]
    text_found:  str  = ""       # OCR text found at this moment (if any)
    audio_text:  str  = ""       # spoken words at this moment (if any)
    frame_idx:   int  = 0
    is_anchor:   bool = False
    tile_id:     str  = "global" # which tile matched (for small objects)


class UnifiedSearchEngine:
    """
    Merges results from all three search signals.

    Initialization:
        engine = UnifiedSearchEngine(
            visual_index    = adve_search_index,
            ocr_extractor   = ocr_extractor,
            audio_indexer   = audio_indexer,   # optional
        )

    Search:
        results = engine.search("backpropagation formula", video_id="lecture")
        # Returns UnifiedResult list, best matches first
    """

    MERGE_WINDOW = 3.0    # seconds: signals within 3s of each other are merged
    MIN_GAP      = 8.0    # seconds: minimum gap between returned results

    def __init__(
        self,
        visual_index,
        ocr_extractor,
        audio_indexer = None,
    ):
        self.visual_index  = visual_index
        self.ocr           = ocr_extractor
        self.audio         = audio_indexer

    # ── Main search ───────────────────────────────────────────────────────

    def search(
        self,
        query:      str,
        video_id:   str,
        k:          int   = 5,
        use_visual: bool  = True,
        use_ocr:    bool  = True,
        use_audio:  bool  = True,
    ) -> List[UnifiedResult]:
        """
        Search using all available signals.
        Returns k results with minimum MIN_GAP seconds between them.
        """
        all_candidates = []

        # ── Signal 1: CLIP Visual ─────────────────────────────────────────
        if use_visual:
            try:
                visual_results = self.visual_index.search_by_text(
                    query, k=30
                )
                visual_results = [
                    r for r in visual_results
                    if r.video_path == video_id
                ]
                for r in visual_results:
                    all_candidates.append(UnifiedResult(
                        video_id   = video_id,
                        timestamp  = r.timestamp,
                        similarity = float(r.similarity),
                        sources    = ["visual"],
                        frame_idx  = getattr(r, "frame_idx", 0),
                        is_anchor  = getattr(r, "is_anchor", False),
                        tile_id    = getattr(r, "camera_id", "global").split("tile_")[-1]
                                     if "tile_" in getattr(r, "camera_id", "")
                                     else "global",
                    ))
            except Exception as e:
                print(f"Visual search error: {e}")

        # ── Signal 2: OCR Text ────────────────────────────────────────────
        if use_ocr and self.ocr:
            try:
                # Both exact and full-text search
                ocr_exact    = self.ocr.search_exact(query, video_id, k=20)
                ocr_fulltext = self.ocr.search_fulltext(query, video_id, k=20)

                for r in ocr_exact + ocr_fulltext:
                    all_candidates.append(UnifiedResult(
                        video_id   = video_id,
                        timestamp  = r["timestamp"],
                        similarity = r["similarity"],
                        sources    = ["ocr"],
                        text_found = r.get("text", ""),
                        frame_idx  = r.get("frame_idx", 0),
                    ))
            except Exception as e:
                print(f"OCR search error: {e}")

        # ── Signal 3: Audio / Whisper ─────────────────────────────────────
        if use_audio and self.audio:
            try:
                audio_results = self.audio.search(query, video_id, k=20)
                for r in audio_results:
                    all_candidates.append(UnifiedResult(
                        video_id   = video_id,
                        timestamp  = r.timestamp,
                        similarity = float(r.similarity),
                        sources    = ["audio"],
                        audio_text = getattr(r, "text", ""),
                    ))
            except Exception as e:
                print(f"Audio search error: {e}")

        if not all_candidates:
            return []

        # ── Merge nearby candidates ───────────────────────────────────────
        merged = self._merge_nearby(all_candidates)

        # ── Sort by score ─────────────────────────────────────────────────
        merged.sort(key=lambda r: r.similarity, reverse=True)

        # ── Temporal deduplication ────────────────────────────────────────
        deduplicated = self._deduplicate(merged, self.MIN_GAP)

        # ── Enrich with OCR context ───────────────────────────────────────
        for r in deduplicated[:k]:
            if self.ocr and not r.text_found:
                r.text_found = self.ocr.get_text_at_timestamp(
                    video_id, r.timestamp, window=2.0
                )

        return deduplicated[:k]

    # ── Merge ─────────────────────────────────────────────────────────────

    def _merge_nearby(self, candidates: List[UnifiedResult]) -> List[UnifiedResult]:
        """
        Merge candidates within MERGE_WINDOW seconds of each other.
        Different signals for the same moment → "both" → score boost.
        """
        merged   = list(candidates)
        combined = []
        used     = set()

        for i, r1 in enumerate(merged):
            if i in used:
                continue

            group = [r1]
            used.add(i)

            for j, r2 in enumerate(merged):
                if j in used or i == j:
                    continue
                if abs(r1.timestamp - r2.timestamp) <= self.MERGE_WINDOW:
                    group.append(r2)
                    used.add(j)

            if len(group) == 1:
                combined.append(r1)
                continue

            # Merge group into one result
            all_sources = list({s for r in group for s in r.sources})
            best_score  = max(r.similarity for r in group)
            source_boost = 1.0 + 0.15 * (len(all_sources) - 1)
            merged_score = min(1.0, best_score * source_boost)

            combined.append(UnifiedResult(
                video_id   = r1.video_id,
                timestamp  = min(r.timestamp for r in group),
                similarity = merged_score,
                sources    = all_sources,
                text_found = next((r.text_found for r in group if r.text_found), ""),
                audio_text = next((r.audio_text for r in group if r.audio_text), ""),
                frame_idx  = r1.frame_idx,
                is_anchor  = any(r.is_anchor for r in group),
            ))

        return combined

    def _deduplicate(
        self,
        results: List[UnifiedResult],
        min_gap: float,
    ) -> List[UnifiedResult]:
        """Keep results at least min_gap seconds apart."""
        kept = []
        for r in results:
            too_close = any(
                abs(r.timestamp - k.timestamp) < min_gap
                for k in kept
            )
            if not too_close:
                kept.append(r)
        return kept

    # ── Format for display ────────────────────────────────────────────────

    @staticmethod
    def source_badge(sources: List[str]) -> str:
        """HTML badge showing which signals matched."""
        colors = {
            "visual": "#1976D2",
            "ocr":    "#E65100",
            "audio":  "#7B1FA2",
        }
        badges = []
        labels = {
            "visual": "👁 Visual",
            "ocr":    "📝 Text",
            "audio":  "🎤 Audio",
        }
        for s in sources:
            c = colors.get(s, "#666")
            l = labels.get(s, s)
            badges.append(
                f'<span style="background:{c};color:white;'
                f'padding:2px 6px;border-radius:4px;'
                f'font-size:11px;margin-right:4px">{l}</span>'
            )
        return "".join(badges)
