"""
ADVE OCR Text Extractor
========================
Extracts readable text from anchor frames using EasyOCR.
Indexes extracted text for exact and semantic search.

What this finds that CLIP misses:
  - Whiteboard equations ("∂L/∂w = Σ x_i * δ_i")
  - Slide text ("Key finding: 23.4% improvement")
  - Signs and labels ("SALE 40% OFF")
  - Subtitles and captions
  - Product text ("iPhone 15 Pro Max ₹1,34,900")
  - Code on screen ("for i in range(n):")
  - Stock tickers ("TSLA +3.2%")
  - Any visible text in any language

Install:
    pip install easyocr

Languages supported: English, Hindi, Tamil, Telugu, and 80+ others.
Set LANGUAGES in config to add regional language support.
"""

import cv2
import numpy as np
import sqlite3
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


# Languages to detect
# Add "hi" for Hindi, "ta" for Tamil, "te" for Telugu
LANGUAGES = ["en"]


@dataclass
class OCRResult:
    text:       str
    confidence: float
    bbox:       tuple    # (x1, y1, x2, y2) in frame pixels
    timestamp:  float
    frame_idx:  int
    video_id:   str


class OCRExtractor:
    """
    Runs EasyOCR on anchor frames to extract all readable text.
    Stores in SQLite for fast text search.
    Embeds text with CLIP for semantic search.

    Two search modes:
      1. Exact search:    "find frame with text '23.4%'"
                          → SQLite LIKE query → instant
      2. Semantic search: "find slide about accuracy metrics"
                          → CLIP text embedding → FAISS search
    """

    def __init__(self, db_path: str, device: str = "cuda"):
        self.device  = device
        self.db_path = db_path
        self._reader = None
        self._init_db()

    # ── Lazy load EasyOCR ────────────────────────────────────────────────

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            use_gpu = (self.device == "cuda")
            self._reader = easyocr.Reader(
                LANGUAGES,
                gpu    = use_gpu,
                verbose = False,
            )
            print(f"EasyOCR loaded (GPU={use_gpu}, langs={LANGUAGES})")
        return self._reader

    # ── Database ─────────────────────────────────────────────────────────

    def _init_db(self):
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ocr_text (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id   TEXT,
                timestamp  REAL,
                frame_idx  INTEGER,
                text       TEXT,
                confidence REAL,
                bbox_json  TEXT
            )
        """)
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_video ON ocr_text(video_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_text ON ocr_text(text)"
        )
        self.db.commit()

    # ── Core extraction ───────────────────────────────────────────────────

    def extract_from_frame(
        self,
        frame:      np.ndarray,
        timestamp:  float,
        frame_idx:  int,
        video_id:   str,
        min_conf:   float = 0.4,
    ) -> List[OCRResult]:
        """
        Run OCR on one frame. Returns list of detected text regions.
        Only returns results with confidence >= min_conf.
        """
        reader  = self._get_reader()
        results = reader.readtext(frame, detail=1, paragraph=False)

        ocr_results = []
        for (bbox_pts, text, confidence) in results:
            if confidence < min_conf or not text.strip():
                continue

            # Convert polygon bbox to rectangle
            pts = np.array(bbox_pts)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())

            ocr_results.append(OCRResult(
                text       = text.strip(),
                confidence = float(confidence),
                bbox       = (x1, y1, x2, y2),
                timestamp  = timestamp,
                frame_idx  = frame_idx,
                video_id   = video_id,
            ))

        return ocr_results

    def store_results(self, results: List[OCRResult]):
        """Store OCR results in SQLite."""
        rows = [
            (
                r.video_id,
                r.timestamp,
                r.frame_idx,
                r.text,
                r.confidence,
                json.dumps(list(r.bbox)),
            )
            for r in results
        ]
        self.db.executemany(
            "INSERT INTO ocr_text VALUES (NULL, ?, ?, ?, ?, ?, ?)", rows
        )
        self.db.commit()

    def index_video(
        self,
        video_path:  str,
        video_id:    str,
        anchor_timestamps: List[float],
        progress_fn  = None,
    ) -> dict:
        """
        Run OCR on all anchor frames of a video.
        anchor_timestamps: list of timestamps where ADVE ran CLIP (anchor frames).
        Only runs OCR on those frames — same adaptive policy as CLIP.
        """
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        total = len(anchor_timestamps)
        found = 0

        for i, timestamp in enumerate(anchor_timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_idx = int(timestamp * fps)
            results   = self.extract_from_frame(frame, timestamp, frame_idx, video_id)

            if results:
                self.store_results(results)
                found += len(results)

            if progress_fn and i % 10 == 0:
                progress_fn(i / total, f"OCR: {i}/{total} frames, {found} text found")

        cap.release()

        print(f"OCR complete: {found} text items from {total} anchor frames")
        return {"text_items_found": found, "frames_processed": total}

    # ── Search ────────────────────────────────────────────────────────────

    def search_exact(
        self,
        query:    str,
        video_id: str,
        k:        int = 10,
    ) -> List[dict]:
        """
        Exact text search using SQL LIKE.
        Fast, finds specific numbers, names, exact phrases.

        Example queries:
          "23.4%"          → finds frame showing "23.4%"
          "backpropagation" → finds frame with that word
          "for i in range"  → finds code on screen
        """
        rows = self.db.execute(
            """
            SELECT timestamp, frame_idx, text, confidence, bbox_json
            FROM ocr_text
            WHERE video_id = ?
              AND LOWER(text) LIKE LOWER(?)
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (video_id, f"%{query}%", k)
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "timestamp":  row[0],
                "frame_idx":  row[1],
                "text":       row[2],
                "confidence": row[3],
                "bbox":       json.loads(row[4]),
                "source":     "ocr_exact",
                "similarity": float(row[3]),
                "video_id":   video_id,
            })

        return results

    def search_fulltext(
        self,
        query:    str,
        video_id: str,
        k:        int = 10,
    ) -> List[dict]:
        """
        Full text search across all OCR results for a video.
        Gets all text at each timestamp, ranks by query word overlap.
        """
        rows = self.db.execute(
            """
            SELECT timestamp, frame_idx,
                   GROUP_CONCAT(text, ' ') as full_text,
                   AVG(confidence)         as avg_conf
            FROM ocr_text
            WHERE video_id = ?
            GROUP BY timestamp
            ORDER BY timestamp
            """,
            (video_id,)
        ).fetchall()

        if not rows:
            return []

        # Score each frame by word overlap with query
        query_words = set(query.lower().split())
        scored = []

        for row in rows:
            frame_text  = (row[2] or "").lower()
            frame_words = set(frame_text.split())
            overlap     = len(query_words & frame_words)

            if overlap > 0:
                score = overlap / len(query_words)
                scored.append({
                    "timestamp":  row[0],
                    "frame_idx":  row[1],
                    "text":       row[2],
                    "source":     "ocr_fulltext",
                    "similarity": float(score),
                    "video_id":   video_id,
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:k]

    def get_text_at_timestamp(
        self,
        video_id:  str,
        timestamp: float,
        window:    float = 2.0,
    ) -> str:
        """Get all text visible at a timestamp (within window seconds)."""
        rows = self.db.execute(
            """
            SELECT text FROM ocr_text
            WHERE video_id = ?
              AND ABS(timestamp - ?) <= ?
            ORDER BY confidence DESC
            """,
            (video_id, timestamp, window)
        ).fetchall()

        return " | ".join(r[0] for r in rows) if rows else ""

    def get_stats(self, video_id: str) -> dict:
        row = self.db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT timestamp) FROM ocr_text WHERE video_id=?",
            (video_id,)
        ).fetchone()
        return {
            "total_text_items": row[0] if row else 0,
            "frames_with_text": row[1] if row else 0,
        }
