import cv2
import numpy as np
from typing import Optional, Tuple


class CameraMotionCompensator:
    """
    Estimates camera motion (homography) between consecutive frames.
    Normalizes object positions to remove camera motion before ΔG computation.
    
    Without this: camera pan → ADVE thinks all objects moved → false anchor refresh
    With this:    camera pan → positions normalized → ΔG stays small → correct behavior
    """

    def __init__(self, max_features: int = 500):
        self.orb       = cv2.ORB_create(max_features)
        self.matcher   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray]         = None  # last homography

    def estimate_homography(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Returns (H, is_camera_motion) where:
          H = 3x3 homography matrix (None if cannot estimate)
          is_camera_motion = True if motion is global (camera) not local (objects)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, False

        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            self.prev_gray = gray
            return None, False

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 8:
            self.prev_gray = gray
            return None, False

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        if H is None:
            self.prev_gray = gray
            return None, False

        # Determine if this is camera motion or object motion
        inlier_ratio = mask.sum() / len(mask)
        is_camera_motion = inlier_ratio > 0.7  # 70%+ inliers = global motion

        self.H = H
        self.prev_gray = gray
        return H, is_camera_motion

    def compensate_position(
        self,
        center: Tuple[float, float],
        H: np.ndarray
    ) -> Tuple[float, float]:
        """Transform a point by the inverse homography to remove camera motion."""
        try:
            H_inv   = np.linalg.inv(H)
            pt      = np.array([[[center[0], center[1]]]], dtype=np.float32)
            pt_comp = cv2.perspectiveTransform(pt, H_inv)
            return float(pt_comp[0, 0, 0]), float(pt_comp[0, 0, 1])
        except np.linalg.LinAlgError:
            return center

    def compensate_graph(self, graph, H: np.ndarray):
        """Apply compensation to all object centers in a SpatialGraph."""
        for obj in graph.objects.values():
            comp_center = self.compensate_position(obj.center, H)
            obj.center  = comp_center
        # Re-build relations with frame size if width/height are available
        # Note: SpatialGraph in v1 has g.width and g.height set
        width = getattr(graph, 'width', None)
        height = getattr(graph, 'height', None)
        if width is not None and height is not None:
            graph.build_relations(width, height)
        else:
            graph.build_relations()
        return graph
