import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


@dataclass
class ObjectState:
    obj_id:     int
    class_name: str
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    center:     Tuple[float, float]
    area:       float
    embedding:  Optional[np.ndarray] = None  # None on delta frames


@dataclass
class Relation:
    distance:   float   # Euclidean pixel distance between centers
    angle:      float   # radians, obj_i → obj_j
    size_ratio: float   # area_j / area_i


class SpatialGraph:
    """
    Directed spatial graph of detected objects in one frame.
    Nodes = objects, Edges = pairwise spatial relations.
    """

    def __init__(self):
        self.objects:   Dict[int, ObjectState]             = {}
        self.relations: Dict[Tuple[int, int], Relation]   = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_object(self, obj: ObjectState) -> None:
        self.objects[obj.obj_id] = obj

    def build_relations(self, width: float = 640.0, height: float = 480.0) -> None:
        ids = sorted(self.objects.keys())
        self.relations = {}
        if len(ids) == 1:
            # Single object case: relate to the frame center (normalized against screen diagonal)
            oid = ids[0]
            o = self.objects[oid]
            cx, cy = width / 2.0, height / 2.0
            dx = o.center[0] - cx
            dy = o.center[1] - cy
            max_dist = np.hypot(cx, cy)
            self.relations[(oid, oid)] = Relation(
                distance=float(np.hypot(dx, dy) / (max_dist + 1e-6)),
                angle=float(np.arctan2(dy, dx)),
                size_ratio=float(o.area / (width * height + 1e-6)),
            )
        else:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    o1, o2 = self.objects[id1], self.objects[id2]

                    dx = o2.center[0] - o1.center[0]
                    dy = o2.center[1] - o1.center[1]

                    self.relations[(id1, id2)] = Relation(
                        distance=float(np.hypot(dx, dy)),
                        angle=float(np.arctan2(dy, dx)),
                        size_ratio=float(o2.area / (o1.area + 1e-6)),
                    )

    # ------------------------------------------------------------------
    # Delta
    # ------------------------------------------------------------------

    def compute_delta(self, other: "SpatialGraph") -> dict:
        """
        Compute structural difference between self (anchor) and other (current).

        Returns
        -------
        dict with keys:
            total_magnitude   – mean normalized distance change (0 = identical)
            relation_deltas   – per-pair change breakdown
            new_objects       – IDs that appeared in `other` but not in `self`
            lost_objects      – IDs in `self` but missing in `other`
        """
        delta: dict = {
            "total_magnitude": 0.0,
            "relation_deltas": {},
            "new_objects": [],
            "lost_objects": [],
        }

        common_pairs: set = set(self.relations) & set(other.relations)
        for pair in common_pairs:
            r_anchor  = self.relations[pair]
            r_current = other.relations[pair]

            d_dist  = abs(r_current.distance   - r_anchor.distance)
            d_angle = abs(r_current.angle       - r_anchor.angle)
            d_size  = abs(r_current.size_ratio  - r_anchor.size_ratio)

            # Normalize distance change relative to anchor distance
            if pair[0] == pair[1]:
                # Single object: d_dist is already normalized to screen diagonal
                norm_dist = d_dist
            else:
                norm_dist = d_dist / (r_anchor.distance + 1e-6)

            delta["relation_deltas"][pair] = {
                "delta_distance":   d_dist,
                "delta_angle":      d_angle,
                "delta_size_ratio": d_size,
                "magnitude":        norm_dist,
            }
            delta["total_magnitude"] += norm_dist

        if common_pairs:
            delta["total_magnitude"] /= len(common_pairs)

        delta["new_objects"]  = [oid for oid in other.objects if oid not in self.objects]
        delta["lost_objects"] = [oid for oid in self.objects  if oid not in other.objects]

        return delta
