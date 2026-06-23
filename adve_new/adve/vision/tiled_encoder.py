"""
ADVE Tiled Encoder
==================
Encodes anchor frames in patches to find small objects that
global CLIP encoding misses.

Standard CLIP encodes a 640x480 frame at 224x224 resolution.
A small text label or tiny logo occupies ~2% of that area.
CLIP's attention never focuses on it.

Tiled encoding divides the frame into patches.
Each patch is encoded separately at full 224x224 resolution.
Small objects that were invisible now occupy 25-50% of a patch.
CLIP finds them.

Usage:
    tiled = TiledEncoder(clip_model, clip_prep, device)
    embeddings = tiled.encode_frame(frame)
    # Returns 5 embeddings: 1 global + 4 tiles
    # Store all 5 in FAISS with the same timestamp
    # Search hits any of them
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple


# Grid configurations
GRIDS = {
    "2x2": (2, 2),   # 4 patches + 1 global = 5 total  ← default
    "3x3": (3, 3),   # 9 patches + 1 global = 10 total ← for very small objects
    "2x1": (2, 1),   # 2 patches + 1 global = 3 total  ← landscape video (16:9)
}


class TiledEncoder:
    """
    Encodes anchor frames as a set of tile embeddings.

    For each anchor frame:
      1 global embedding  (full scene context)
      N tile embeddings   (fine-grained local details)

    All stored in FAISS with the same timestamp but
    different camera_id tags for filtering.

    Benefits:
      - Small objects (text on whiteboard, tiny logos) become searchable
      - Fine-grained spatial queries ("find the equation in the top-right")
      - No additional model required — reuses CLIP
    """

    def __init__(self, clip_model, clip_prep, device: str = "cuda"):
        self.clip_model = clip_model
        self.clip_prep  = clip_prep
        self.device     = device

    def _embed(self, image: np.ndarray) -> np.ndarray:
        """Embed a single image (BGR numpy array)."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            t   = self.clip_prep(pil).unsqueeze(0).to(self.device)
            emb = self.clip_model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten().astype(np.float32)

    def extract_tiles(
        self,
        frame:  np.ndarray,
        grid:   str = "2x2",
        overlap: float = 0.1,
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Split frame into tiles with optional overlap.

        Returns list of (tile_image, tile_label) tuples.
        tile_label: "global", "tile_0_0", "tile_0_1", etc.

        Overlap prevents missing objects at tile boundaries.
        overlap=0.1 means each tile extends 10% into adjacent tiles.
        """
        h, w = frame.shape[:2]
        rows, cols = GRIDS.get(grid, (2, 2))

        tile_h = h // rows
        tile_w = w // cols
        pad_h  = int(tile_h * overlap)
        pad_w  = int(tile_w * overlap)

        tiles = [("global", frame)]  # always include global

        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * tile_h - pad_h)
                y2 = min(h, (r + 1) * tile_h + pad_h)
                x1 = max(0, c * tile_w - pad_w)
                x2 = min(w, (c + 1) * tile_w + pad_w)

                tile = frame[y1:y2, x1:x2]
                if tile.size > 0:
                    tiles.append((f"tile_{r}_{c}", tile))

        return tiles

    def encode_frame(
        self,
        frame:  np.ndarray,
        grid:   str = "2x2",
    ) -> List[dict]:
        """
        Encode a frame into global + tile embeddings.

        Returns list of dicts:
          {
            "embedding": np.ndarray (512-d),
            "tile_id":   str ("global", "tile_0_0", etc.),
            "bbox":      (x1, y1, x2, y2) in original frame coords
          }

        Store all in FAISS — search hits any tile.
        """
        h, w = frame.shape[:2]
        rows, cols = GRIDS.get(grid, (2, 2))
        tile_h = h // rows
        tile_w = w // cols

        results = []

        # Global embedding
        results.append({
            "embedding": self._embed(frame),
            "tile_id":   "global",
            "bbox":      (0, 0, w, h),
        })

        # Tile embeddings
        for r in range(rows):
            for c in range(cols):
                y1 = r * tile_h
                y2 = min(h, (r + 1) * tile_h)
                x1 = c * tile_w
                x2 = min(w, (c + 1) * tile_w)

                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                results.append({
                    "embedding": self._embed(tile),
                    "tile_id":   f"tile_{r}_{c}",
                    "bbox":      (x1, y1, x2, y2),
                })

        return results

    def encode_batch(
        self,
        frames:     List[np.ndarray],
        grid:       str = "2x2",
        batch_size: int = 8,
    ) -> List[List[dict]]:
        """Encode multiple frames efficiently."""
        return [self.encode_frame(f, grid) for f in frames]
