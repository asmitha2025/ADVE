import numpy as np
import faiss
import json
import sqlite3
import cv2
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
        self.index_dir.mkdir(exist_ok=True, parents=True)
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
        if not records:
            return
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

        query_vec = text_emb.cpu().numpy().astype(np.float32).flatten()
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

        query_vec = emb.cpu().numpy().astype(np.float32).flatten()
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
