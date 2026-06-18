import numpy as np
import cv2
from ultralytics import YOLO
from typing import Tuple, Optional

from spatial_graph import SpatialGraph, ObjectState


class DeltaTracker:
    """
    Processes delta (non-keyframe) frames.

    DOES NOT run CLIP. Only:
      1. Runs YOLO + ByteTrack to get updated bounding boxes.
      2. Transfers embeddings from anchor graph for matched track IDs.
      3. Returns the updated SpatialGraph + ΔG dict.

    This is the core cost-saving path.
    """

    def __init__(self, yolo: YOLO, device: str = 'cpu'):
        self.yolo   = yolo  # Same shared YOLO instance as AnchorProcessor
        self.device = device

    def track(
        self,
        frame: np.ndarray,
        anchor_graph: SpatialGraph,
        homography: Optional[np.ndarray] = None
    ) -> Tuple[SpatialGraph, dict]:
        """
        Parameters
        ----------
        frame        : current BGR frame
        anchor_graph : most recent anchor SpatialGraph (has embeddings)
        homography   : estimated camera homography matrix (anchor -> current)

        Returns
        -------
        current_graph : SpatialGraph with reused embeddings for tracked objects
        delta         : output of anchor_graph.compute_delta(current_graph)
        """
        results = self.yolo.track(frame, persist=True, verbose=False, device=self.device)[0]
        current = SpatialGraph()

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                if box.id is None:
                    continue

                obj_id     = int(box.id[0])
                class_name = self.yolo.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Reuse embedding from anchor for matching track ID
                # None if this is a new object (Branch 2)
                embedding = (
                    anchor_graph.objects[obj_id].embedding
                    if obj_id in anchor_graph.objects
                    else None
                )

                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                if homography is not None:
                    try:
                        _, inv_H = cv2.invert(homography)
                        pt = np.array([center[0], center[1], 1.0]).reshape(3, 1)
                        warped = np.dot(inv_H, pt)
                        if abs(warped[2, 0]) > 1e-6:
                            center = (float(warped[0, 0] / warped[2, 0]), float(warped[1, 0] / warped[2, 0]))
                    except Exception:
                        pass

                current.add_object(ObjectState(
                    obj_id=obj_id,
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=float((x2 - x1) * (y2 - y1)),
                    embedding=embedding,
                ))

        current.build_relations(frame.shape[1], frame.shape[0])
        delta = anchor_graph.compute_delta(current)

        return current, delta
