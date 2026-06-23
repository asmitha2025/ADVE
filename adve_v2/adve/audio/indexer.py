"""
ADVE Audio Indexer
==================
Transcribes video audio using Whisper and indexes each sentence
as a searchable embedding alongside visual CLIP embeddings.

Result: users can search by what was SAID or what was SHOWN.
Both visual and audio results returned together, ranked by relevance.

Install:
    pip install openai-whisper

Usage:
    from adve.audio.indexer import AudioIndexer
    audio_idx = AudioIndexer(search_index)
    audio_idx.index_video("lecture.mp4", "lecture_id")
    results = audio_idx.search("gradient descent")
"""

import os
import json
import sqlite3
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioSearchResult:
    video_path:  str
    timestamp:   float
    text:        str          # the spoken words at this moment
    similarity:  float
    source:      str = "audio"


class AudioIndexer:
    """
    Transcribes video audio with Whisper.
    Embeds each sentence with CLIP text encoder.
    Stores in the existing FAISS + SQLite index alongside visual embeddings.

    This gives multimodal search:
      - Visual: "show me a whiteboard with equations"
      - Audio:  "find where they explain backpropagation"
      - Both:   combined results, ranked together
    """

    def __init__(self, search_index, device: str = "cuda", clip_model=None, clip_prep=None):
        self.search_index = search_index
        self.device       = device
        self._whisper     = None
        self._clip_model  = clip_model
        self._clip_prep   = clip_prep
        if clip_model is not None:
            self._clip_device = next(clip_model.parameters()).device

    # ── Lazy loading ────────────────────────────────────────────────────

    def _get_whisper(self):
        if self._whisper is None:
            import whisper
            # Use "base" model — fast, good enough for search
            # Upgrade to "small" or "medium" for better accuracy
            self._whisper = whisper.load_model("base", device=self.device)
            print("Whisper loaded: base model")
        return self._whisper

    def _get_clip(self):
        if self._clip_model is None:
            import torch
            from adve.core.config import Config
            config = Config()
            device = self.device if torch.cuda.is_available() else "cpu"
            from adve.core.clip_loader import load_clip_cached
            self._clip_model, self._clip_prep = load_clip_cached(config.CLIP_MODEL, device=device)
            self._clip_device = device
        return self._clip_model, self._clip_prep

    # ── Audio extraction ────────────────────────────────────────────────

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video as 16kHz mono WAV for Whisper."""
        wav_path = video_path.replace(".mp4", "_audio.wav").replace(".webm", "_audio.wav")

        if os.path.exists(wav_path):
            return wav_path

        subprocess.run([
            "ffmpeg", "-y",
            "-i",        video_path,
            "-ar",       "16000",      # 16kHz required by Whisper
            "-ac",       "1",          # mono
            "-c:a",      "pcm_s16le",  # WAV format
            wav_path,
        ], capture_output=True, check=True)

        return wav_path

    # ── Transcription ───────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> list:
        """
        Transcribe audio. Returns list of segments with timestamps.

        Each segment:
          {"start": 4.2, "end": 7.8, "text": "the key insight is..."}
        """
        model    = self._get_whisper()
        result   = model.transcribe(
            audio_path,
            verbose         = False,
            word_timestamps = False,
        )
        return result.get("segments", [])

    # ── Text embedding with CLIP ─────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLIP text encoder. Returns 512-d unit vector."""
        import torch, clip
        clip_model, _ = self._get_clip()

        with torch.no_grad():
            tokens = clip.tokenize(
                [text[:77]],     # CLIP max token length is 77
                truncate=True,
            ).to(self._clip_device)

            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)

    # ── Index ────────────────────────────────────────────────────────────

    def index_video(
        self,
        video_path: str,
        video_id:   str,
        progress_fn = None,
    ) -> dict:
        """
        Transcribe video audio and index each sentence as a searchable embedding.

        Storage convention:
          camera_id = "{video_id} [AUDIO]"
          This lets us distinguish audio from visual results in search.
        """
        print(f"Extracting audio: {video_path}")
        audio_path = self.extract_audio(video_path)

        print("Transcribing with Whisper...")
        segments   = self.transcribe(audio_path)

        if not segments:
            print("No speech detected in video.")
            return {"segments_indexed": 0}

        print(f"Transcribed {len(segments)} segments. Embedding...")

        batch  = []
        camera = f"{video_id} [AUDIO]"

        for i, seg in enumerate(segments):
            text  = seg["text"].strip()
            start = float(seg["start"])

            if not text:
                continue

            # Embed the spoken text using CLIP text encoder
            emb = self._embed_text(text)

            batch.append({
                "video_path": video_id,
                "camera_id":  camera,
                "timestamp":  start,
                "frame_idx":  int(start * 30),  # approximate frame
                "embedding":  emb,
                "is_anchor":  True,
                "text":       text,
            })

            if progress_fn and i % 20 == 0:
                progress_fn(i / len(segments), f"Indexed {i}/{len(segments)} audio segments")

        # Store text in SQLite alongside embeddings
        # Add text column if it doesn't exist
        try:
            self.search_index.db.execute("ALTER TABLE embeddings ADD COLUMN text TEXT")
            self.search_index.db.commit()
        except Exception:
            pass  # column already exists

        # Write to index
        self.search_index.add_batch(batch, text_col=True)
        self.search_index.save()

        # Clean up temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"Audio indexed: {len(batch)} segments from '{video_id}'")
        return {"segments_indexed": len(batch), "video_id": video_id}

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query:    str,
        video_id: str,
        k:        int = 10,
    ) -> list:
        """
        Search audio transcriptions.
        Embeds query as text, searches against text-embedded segments.
        """
        query_vec = self._embed_text(query)

        # Search only audio entries for this video
        rows = self.search_index.db.execute(
            "SELECT id, timestamp, text FROM embeddings "
            "WHERE camera_id=?",
            (f"{video_id} [AUDIO]",)
        ).fetchall()

        if not rows:
            return []

        import faiss
        ids  = [r[0] - 1 for r in rows]
        txts = {r[0] - 1: r[2] for r in rows}  # faiss_id → text
        tss  = {r[0] - 1: r[1] for r in rows}  # faiss_id → timestamp

        # Build sub-index from audio entries
        sub_embs = np.vstack([
            self.search_index.faiss_index.reconstruct(i)
            for i in ids
            if 0 <= i < self.search_index.faiss_index.ntotal
        ]).astype(np.float32)

        sub_index = faiss.IndexFlatIP(self.search_index.dim)
        sub_index.add(sub_embs)

        actual_k       = min(k, sub_index.ntotal)
        scores, indices = sub_index.search(query_vec.reshape(1, -1), actual_k)

        results = []
        for score, sub_idx in zip(scores[0], indices[0]):
            if sub_idx < 0 or sub_idx >= len(ids):
                continue
            faiss_id = ids[sub_idx]
            results.append(AudioSearchResult(
                video_path = video_id,
                timestamp  = tss.get(faiss_id, 0.0),
                text       = txts.get(faiss_id, ""),
                similarity = float(score),
                source     = "audio",
            ))

        return results
