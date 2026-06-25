import numpy as np
import cv2
import torch
import clip
from PIL import Image
from ultralytics import YOLO
from typing import Tuple

from adve.core.config import Config
from adve.core.spatial_graph import SpatialGraph, ObjectState


class AnchorProcessor:
    """
    Processes anchor (keyframe) frames.
    Runs full CLIP on the whole frame AND on each detected object's RoI.
    Builds a SpatialGraph with per-object embeddings.
    """

    def __init__(self, config: Config, yolo: YOLO, clip_model=None, clip_preprocess=None):
        self.config = config
        self.device = getattr(config, "CLIP_DEVICE", config.DEVICE)
        self.yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        self.yolo   = yolo  # Shared YOLO instance (preserves ByteTrack ID state)

        if clip_model is not None and clip_preprocess is not None:
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess
            self.clip_dim = self.clip_model.visual.output_dim
        else:
            from adve.core.clip_loader import load_clip_cached
            self.clip_model, self.clip_preprocess = load_clip_cached(
                config.CLIP_MODEL, device=self.device
            )
            self.clip_dim = self.clip_model.visual.output_dim

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[SpatialGraph, np.ndarray]:
        """
        Parameters
        ----------
        frame : BGR numpy array

        Returns
        -------
        graph             : SpatialGraph with embeddings on each ObjectState
        frame_embedding   : clip_dim-d CLIP embedding of full frame (normalised)
        """
        frame_embedding = self._embed(frame)

        imgsz = getattr(self.config, "YOLO_IMGSZ", 320)
        results = self.yolo.track(frame, imgsz=imgsz, persist=True, verbose=False, device=self.yolo_device)[0]
        graph   = SpatialGraph()

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                if box.id is None:
                    continue

                obj_id     = int(box.id[0])
                class_name = self.yolo.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Clamp to frame bounds
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                obj_embedding = self._embed_object(roi)

                # Compute histogram for appearance check (Improvement 6)
                hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8],
                                    [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                area   = float((x2 - x1) * (y2 - y1))

                graph.add_object(ObjectState(
                    obj_id=obj_id,
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area,
                    embedding=obj_embedding,
                    appearance_hist=hist,
                ))

        graph.build_relations(frame.shape[1], frame.shape[0])
        return graph, frame_embedding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """Public method used by Validator for ground-truth computation."""
        return self._embed(frame)

    def _embed_object(self, roi: np.ndarray) -> np.ndarray:
        """Embed an object crop using the standard CLIP visual encoder."""
        return self._embed(roi)

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros(self.clip_dim, dtype=np.float32)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            emb    = self.clip_model.encode_image(tensor)
            emb    = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)
