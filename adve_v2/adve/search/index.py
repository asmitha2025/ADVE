import os
import numpy as np
import faiss
import json
import sqlite3
import cv2
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


def normalize_video_path(video_path: str) -> str:
    """Normalize a video path to standard format (absolute, normalized path)."""
    if not video_path:
        return ""
    return os.path.normpath(os.path.abspath(video_path))


@dataclass
class SearchResult:
    video_path:  str
    camera_id:   str
    timestamp:   float
    frame_idx:   int
    similarity:  float
    is_anchor:   bool = False
    thumbnail:   Optional[str] = None  # base64 jpg
    text:        Optional[str] = ""
    objects:     Optional[List[str]] = None


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

    def __init__(self, index_dir: str, dim: Optional[int] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        if dim is None:
            try:
                from adve.core.config import Config
                config = Config()
                if "ViT-L/14" in config.CLIP_MODEL:
                    dim = 768
                elif "RN50x" in config.CLIP_MODEL or "RN101" in config.CLIP_MODEL:
                    dim = 512
                elif "RN50" in config.CLIP_MODEL:
                    dim = 1024
                else:
                    dim = 512
            except ImportError:
                dim = 512
                
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
                is_anchor  INTEGER,
                text       TEXT
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                timestamp  REAL,
                text       TEXT
            )
        """)
        try:
            self.db.execute("ALTER TABLE embeddings ADD COLUMN text TEXT")
        except Exception:
            pass
        self.db.commit()

    def add_transcripts(self, video_path: str, segments: List[Dict]):
        """Add transcribed speech segments to index."""
        if not segments:
            return
        self.db.executemany(
            "INSERT INTO transcripts VALUES (NULL, ?, ?, ?)",
            [(normalize_video_path(video_path), s["timestamp"], s["text"]) for s in segments]
        )
        self.db.commit()

    def add(
        self,
        video_path: str,
        camera_id:  str,
        timestamp:  float,
        frame_idx:  int,
        embedding:  np.ndarray,
        is_anchor:  bool = False,
        text:       str = "",
    ):
        # Normalize to unit sphere
        emb = embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        self.faiss_index.add(emb.reshape(1, -1))

        self.db.execute(
            "INSERT INTO embeddings (video_path, camera_id, timestamp, frame_idx, is_anchor, text) VALUES (?, ?, ?, ?, ?, ?)",
            (normalize_video_path(video_path), camera_id, timestamp, frame_idx, int(is_anchor), text)
        )
        self.db.commit()
        self.save()

    def add_batch(self, records: List[Dict], text_col: bool = True):
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
            "INSERT INTO embeddings (video_path, camera_id, timestamp, frame_idx, is_anchor, text) VALUES (?, ?, ?, ?, ?, ?)",
            [(normalize_video_path(r["video_path"]), r["camera_id"], r["timestamp"],
              r["frame_idx"], int(r.get("is_anchor", False)), r.get("text", ""))
             for r in records]
        )
        self.db.commit()
        self.save()

    def _expand_query_via_groq(self, query: str) -> str:
        """Expand natural language query using Groq LLM for better CLIP matching (e.g. negation, count translation)."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return query
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            prompt = (
                "You are a search query expansion assistant for a CLIP-based video retrieval system.\n"
                "Your job is to translate complex queries into descriptive visual scenes that CLIP can match better.\n"
                "Handle negation, counts, action descriptions, and objects by describing what is physically visible in the frame.\n"
                "For example:\n"
                "- 'street with no cars' -> 'empty street, clear asphalt, no traffic, quiet road'\n"
                "- 'three people' -> 'trio of people, three individuals, group of three persons'\n"
                "- 'a classroom with no teacher' -> 'classroom, empty teacher desk, students in chairs, no teacher'\n"
                "\n"
                "Respond ONLY with the expanded query as a comma-separated list of visual descriptions.\n"
                "Do not include explanation, prefix, or markdown. Keep it concise (maximum 15 words).\n\n"
                f"Original query: '{query}'\n"
                "Expanded query:"
            )
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.1
            )
            expanded = response.choices[0].message.content.strip().strip("'\"")
            print(f"[Groq Query Expansion] '{query}' -> '{expanded}'")
            return expanded
        except Exception as e:
            print(f"[Groq Query Expansion] Warning: failed to expand query: {e}")
            return query

    def search_by_text(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search video content using natural language (visual) and speech (audio)."""
        # Expand query via Groq if key is available
        expanded_query = self._expand_query_via_groq(query) if os.environ.get("GROQ_API_KEY") else query

        import torch, clip
        if self._clip_model is None:
            try:
                from adve.core.config import Config
                config = Config()
                clip_model_name = config.CLIP_MODEL
                device = config.CLIP_DEVICE
            except ImportError:
                clip_model_name = "ViT-B/32"
                device = "cpu"
            from adve.core.clip_loader import load_clip_cached
            self._clip_model, self._clip_prep = load_clip_cached(clip_model_name, device=device)

        device = next(self._clip_model.parameters()).device
        with torch.no_grad():
            tokens = clip.tokenize([expanded_query]).to(device)
            text_emb = self._clip_model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        query_vec = text_emb.cpu().numpy().astype(np.float32).flatten()
        
        # 1. Search visual features via FAISS
        visual_results = self._search(query_vec, k, query=query)
        
        # 2. Search speech transcripts via SQLite using keyword matching
        audio_results = []
        try:
            query_clean = query.lower().strip()
            query_words = [w for w in query_clean.split() if len(w) > 1]
            
            stopwords = {
                "the", "a", "of", "in", "on", "is", "at", "which", "to", "for", 
                "with", "and", "or", "an", "this", "that", "it", "from", "by", 
                "are", "was", "were", "we", "you", "i", "he", "she", "they", 
                "them", "us", "about", "how", "what", "why", "where", "when"
            }
            filtered_words = [w for w in query_words if w not in stopwords]
            if not filtered_words:
                filtered_words = query_words

            # Fetch transcripts containing any of the keywords
            if filtered_words:
                sql_conditions = " OR ".join(["text LIKE ?" for _ in filtered_words])
                params = [f"%{w}%" for w in filtered_words]
                cursor = self.db.execute(
                    f"SELECT video_path, timestamp, text FROM transcripts WHERE {sql_conditions}",
                    params
                )
                rows = cursor.fetchall()
            else:
                rows = []

            for row in rows:
                v_path, ts, text = row
                text_lower = text.lower()
                
                # Calculate word overlap scoring
                if query_clean in text_lower:
                    score = 0.99
                else:
                    matches = sum(1 for w in filtered_words if w in text_lower)
                    match_ratio = matches / len(filtered_words) if filtered_words else 0.0
                    
                    if match_ratio == 0:
                        continue
                    
                    # Score scaled from 0.70 to 0.95
                    score = 0.70 + 0.25 * match_ratio

                # Find matching closest frame index in embeddings
                frame_row = self.db.execute(
                    "SELECT frame_idx, camera_id FROM embeddings WHERE video_path = ? ORDER BY ABS(timestamp - ?) LIMIT 1",
                    (v_path, ts)
                ).fetchone()
                
                frame_idx = frame_row[0] if frame_row else int(ts * 30)
                camera_id = frame_row[1] if frame_row else "stream"
                
                audio_results.append(SearchResult(
                    video_path = normalize_video_path(v_path),
                    camera_id = f"{camera_id} (Speech: \"{text}\")",
                    timestamp = ts,
                    frame_idx = frame_idx,
                    similarity = score,
                ))
        except Exception as e:
            print(f"[ADVESearchIndex] Transcript search error: {e}")

        # Combine results, sort by similarity
        combined = visual_results + audio_results
        combined.sort(key=lambda x: -x.similarity)
        return combined[:k]

    def search_by_image(
        self, image: np.ndarray, k: int = 10
    ) -> List[SearchResult]:
        """Search using an image query — find similar scenes."""
        import torch, clip
        from PIL import Image

        if self._clip_model is None:
            try:
                from adve.core.config import Config
                config = Config()
                clip_model_name = config.CLIP_MODEL
                device = config.CLIP_DEVICE
            except ImportError:
                clip_model_name = "ViT-B/32"
                device = "cpu"
            from adve.core.clip_loader import load_clip_cached
            self._clip_model, self._clip_prep = load_clip_cached(clip_model_name, device=device)

        device = next(self._clip_model.parameters()).device
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            t   = self._clip_prep(pil).unsqueeze(0).to(device)
            emb = self._clip_model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        query_vec = emb.cpu().numpy().astype(np.float32).flatten()
        return self._search(query_vec, k)

    def _search(self, query_vec: np.ndarray, k: int, query: Optional[str] = None) -> List[SearchResult]:
        if self.faiss_index.ntotal == 0:
            return []

        # Get query words for object matching Stage 2
        query_words = []
        if query:
            query_words = [w.lower().strip(",.!?\"'") for w in query.split()]
            query_words = [w for w in query_words if len(w) > 2]

        k = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.db.execute(
                "SELECT video_path, camera_id, timestamp, frame_idx, is_anchor, text "
                "FROM embeddings WHERE id=?", (int(idx) + 1,)
            ).fetchone()

            if row:
                video_path = normalize_video_path(row[0])
                camera_id  = row[1]
                timestamp  = row[2]
                frame_idx  = row[3]
                is_anchor  = bool(row[4])
                base_sim   = float(score)
                objects_str = row[5] if row[5] else ""

                # Stage 2: Boost similarity if query contains any of the detected object class names
                boost = 0.0
                detected_objs = []
                if objects_str:
                    detected_objs = [obj.strip().lower() for obj in objects_str.split(",") if obj.strip()]
                    if query_words:
                        # Check overlap
                        matched_objs = [w for w in query_words if w in detected_objs]
                        if matched_objs:
                            # Apply a similarity boost (e.g. 0.08 per matched object, max boost 0.15)
                            boost = min(0.15, 0.08 * len(matched_objs))

                final_sim = min(1.0, base_sim + boost)

                results.append(SearchResult(
                    video_path = video_path,
                    camera_id  = camera_id,
                    timestamp  = timestamp,
                    frame_idx  = frame_idx,
                    similarity = final_sim,
                    is_anchor  = is_anchor,
                    text       = objects_str,
                    objects    = detected_objs,
                ))

        return results

    def save(self):
        faiss.write_index(
            self.faiss_index, str(self.index_dir / "embeddings.faiss")
        )

    def clear(self):
        """Clear all embeddings and transcripts from the index and database."""
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.db.execute("DELETE FROM embeddings")
        self.db.execute("DELETE FROM transcripts")
        self.db.execute("DELETE FROM sqlite_sequence WHERE name IN ('embeddings', 'transcripts')")
        self.db.commit()
        self.save()


    def stats(self) -> dict:
        count = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        anchors = self.db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE is_anchor=1"
        ).fetchone()[0]
        
        # Get unique videos/cameras
        cursor = self.db.execute("SELECT DISTINCT video_path, camera_id FROM embeddings")
        videos = [{"video_path": row[0], "camera_id": row[1]} for row in cursor.fetchall()]
        
        return {
            "total_embeddings": count,
            "anchor_frames":    anchors,
            "delta_frames":     count - anchors,
            "index_size_mb":    round(
                (self.index_dir / "embeddings.faiss").stat().st_size / 1e6, 2
            ) if (self.index_dir / "embeddings.faiss").exists() else 0,
            "indexed_videos":   videos,
        }
