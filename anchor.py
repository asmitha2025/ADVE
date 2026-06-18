import numpy as np
import cv2
import torch
import clip
from PIL import Image
from ultralytics import YOLO
from typing import Tuple

from config import Config
from spatial_graph import SpatialGraph, ObjectState


class AnchorProcessor:
    """
    Processes anchor (keyframe) frames.
    Runs full CLIP on the whole frame AND on each detected object's RoI.
    Builds a SpatialGraph with per-object embeddings.
    """

    def __init__(self, config: Config, yolo: YOLO):
        self.config = config
        self.device = config.DEVICE
        self.yolo   = yolo  # Shared YOLO instance (preserves ByteTrack ID state)

        self.clip_model, self.clip_preprocess = clip.load(
            config.CLIP_MODEL, device=self.device
        )
        self.clip_model.eval()

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
        frame_embedding   : 512-d CLIP embedding of full frame (normalised)
        """
        frame_embedding = self._embed(frame)

        results = self.yolo.track(frame, persist=True, verbose=False, device=self.device)[0]
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
                obj_embedding = self._embed(roi)

                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                area   = float((x2 - x1) * (y2 - y1))

                graph.add_object(ObjectState(
                    obj_id=obj_id,
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area,
                    embedding=obj_embedding,
                ))

        graph.build_relations(frame.shape[1], frame.shape[0])
        return graph, frame_embedding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """Public method used by Validator for ground-truth computation."""
        return self._embed(frame)

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros(512, dtype=np.float32)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            emb    = self.clip_model.encode_image(tensor)
            emb    = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)
