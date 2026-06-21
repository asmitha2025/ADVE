import cv2
import numpy as np
import os
import time
from typing import Optional
from ultralytics import YOLO
from adve.core.frame_filter import FrameFilter

from adve.core.config import Config
from adve.core.spatial_graph import SpatialGraph
from adve.core.anchor import AnchorProcessor
from adve.core.tracker import DeltaTracker
from adve.core.reconstructor import EmbeddingReconstructor
from adve.core.validator import Validator


class ADVEPipeline:
    """
    ADVE — Anchor-Delta Video Embedding

    Orchestrates the full pipeline:
      1. Decide: anchor frame or delta frame
      2. Anchor → full CLIP + YOLO + SpatialGraph build
      3. Delta  → ByteTrack only + embedding reconstruction
      4. Validate per-frame cosine similarity vs ground truth
    """

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Single shared YOLO instance — preserves ByteTrack ID state across frames
        self.yolo = YOLO(config.YOLO_MODEL)
        yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        self.yolo.to(yolo_device)
        
        # Enable FP16 (half precision) for YOLO on CUDA
        yolo_half = getattr(config, "YOLO_HALF", True)
        if yolo_half and yolo_device == "cuda":
            try:
                self.yolo.model.half()
                print("[ADVEPipeline] Enabled FP16 (Half Precision) for YOLOv8 tracking on GPU.")
            except Exception as e:
                print(f"[ADVEPipeline] Warning: Failed to convert YOLO model to FP16: {e}")

        self.anchor_proc     = AnchorProcessor(config, self.yolo)
        yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        yolo_imgsz = getattr(config, "YOLO_IMGSZ", 320)
        self.delta_tracker   = DeltaTracker(self.yolo, device=yolo_device, imgsz=yolo_imgsz)
        self.reconstructor   = EmbeddingReconstructor(config.MLP_MODEL_PATH)
        self.validator       = Validator(config)

        # Live state
        self.anchor_graph:     Optional[SpatialGraph] = None
        self.anchor_embedding: Optional[np.ndarray]   = None
        self.anchor_frame:     Optional[np.ndarray]   = None
        self.frames_since_anchor: int = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.force_refresh:    bool = False
        motion_threshold = getattr(config, "MOTION_THRESHOLD", 0.02)
        self.frame_filter = FrameFilter(motion_threshold=motion_threshold)

    # ------------------------------------------------------------------
    # Anchor decision logic
    # ------------------------------------------------------------------

    def _needs_anchor(self, delta: dict, appearance_delta: float) -> bool:
        return (
            delta["total_magnitude"]    > self.config.SPATIAL_THRESHOLD  or
            appearance_delta            > self.config.APPEARANCE_THRESHOLD or
            self.frames_since_anchor   >= self.config.MAX_DELTA_FRAMES    or
            len(delta["new_objects"])   > 0
        )

    def _appearance_delta(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Fast histogram-based appearance change score."""
        def hist(f):
            h = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
            return cv2.normalize(h, h).flatten()

        corr = cv2.compareHist(hist(f1), hist(f2), cv2.HISTCMP_CORREL)
        return float(1.0 - corr)   # 0 = identical, 1 = completely different

    def _estimate_homography(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        try:
            # Resize images to a max dimension of 480px to speed up ORB computation on CPU
            h, w = img1.shape[:2]
            scale = 480.0 / max(h, w)
            if scale < 1.0:
                img1_small = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
                img2_small = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
            else:
                img1_small = img1
                img2_small = img2
                scale = 1.0

            gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=300)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) < 8:
                return None
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Scale coordinates back to original size before estimating homography
            if scale < 1.0:
                src_pts = src_pts / scale
                dst_pts = dst_pts / scale
                
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int, no_validation: bool = True) -> dict:
        # --- Motion filter ---
        if self.anchor_graph is not None:
            has_motion, score = self.frame_filter.has_motion(frame)
            if not has_motion:
                self.prev_frame = frame.copy()
                self.frames_since_anchor += 1
                
                # Log in validator to keep tracking continuous
                sim = self.validator.log(
                    frame_idx       = frame_idx,
                    reconstructed   = self.anchor_embedding,
                    ground_truth    = None if no_validation else self.anchor_proc.embed_frame(frame),
                    is_anchor       = False,
                    delta_magnitude = 0.0,
                    encoder_called  = False,
                )
                
                return {
                    "embedding":       self.anchor_embedding,
                    "is_anchor":       False,
                    "encoder_called":  False,
                    "delta_magnitude": 0.0,
                    "appearance_delta": 0.0,
                    "cosine_sim":      sim,
                    "frame_idx":       frame_idx,
                    "yolo_skipped":    True,
                }

        is_anchor      = False
        encoder_called = False
        delta_magnitude = 0.0
        appearance_delta = 0.0
        current_graph  = None
        delta          = {"total_magnitude": 0.0, "new_objects": [], "lost_objects": []}

        # --- Appearance delta ---
        if self.prev_frame is not None:
            appearance_delta = self._appearance_delta(self.prev_frame, frame)

        # --- Decide frame type ---
        if self.anchor_graph is None or self.force_refresh:
            refresh = True
            self.force_refresh = False
        else:
            homography = self._estimate_homography(self.anchor_frame, frame)
            current_graph, delta = self.delta_tracker.track(
                frame, self.anchor_graph, homography=homography
            )
            delta_magnitude = delta["total_magnitude"]
            refresh = self._needs_anchor(delta, appearance_delta)

        # --- Process ---
        if refresh:
            # ── ANCHOR FRAME ──────────────────────────────────────
            self.anchor_frame = frame.copy()
            self.anchor_graph, self.anchor_embedding = self.anchor_proc.process(frame)
            self.frames_since_anchor = 0
            is_anchor      = True
            encoder_called = True

            reconstructed     = self.anchor_embedding
            ground_truth      = self.anchor_embedding    # trivially 1.0

        else:
            # ── DELTA FRAME ───────────────────────────────────────
            reconstructed = self.reconstructor.reconstruct(
                self.anchor_graph,
                current_graph,
                delta,
                self.anchor_embedding,
            )

            # Ground truth: full CLIP (only for validation — normally skipped)
            ground_truth = None if no_validation else self.anchor_proc.embed_frame(frame)
            self.frames_since_anchor += 1

            # Check if current frame was empty, schedule refresh if anchor was not empty
            if len(current_graph.objects) == 0 and len(self.anchor_graph.objects) > 0:
                self.force_refresh = True

        self.prev_frame = frame.copy()

        # --- Validate ---
        sim = self.validator.log(
            frame_idx       = frame_idx,
            reconstructed   = reconstructed,
            ground_truth    = ground_truth,
            is_anchor       = is_anchor,
            delta_magnitude = delta_magnitude,
            encoder_called  = encoder_called,
        )

        return {
            "embedding":       reconstructed,
            "is_anchor":       is_anchor,
            "encoder_called":  encoder_called,
            "delta_magnitude": delta_magnitude,
            "appearance_delta": appearance_delta,
            "cosine_sim":      sim,
            "frame_idx":       frame_idx,
        }

    def process_video(self, video_path: str, no_validation: bool = False) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n{'='*55}")
        print(f"  ADVE Pipeline  |  Device: {self.config.DEVICE.upper()}")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Frames: {total_vid_frames}  |  FPS: {total_fps:.1f}")
        print(f"{'='*55}")

        frame_idx = 0
        t_start   = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            res = self.process_frame(frame, frame_idx, no_validation=no_validation)

            if frame_idx % 15 == 0:
                tag = "ANCHOR" if res["is_anchor"] else "DELTA "
                print(
                    f"  [{frame_idx:>5}] {tag}  "
                    f"sim={res['cosine_sim']:.4f}  ΔG={res['delta_magnitude']:.4f}  "
                    f"Δapp={res['appearance_delta']:.4f}"
                )

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t_start

        # --- Results ---
        summary = self.validator.summarize()
        summary["elapsed_sec"] = round(elapsed, 2)
        summary["effective_fps"] = round(frame_idx / elapsed, 1)

        self.validator.plot(
            os.path.join(self.config.OUTPUT_DIR, "adve_results.png")
        )
        self.validator.save_json(
            os.path.join(self.config.OUTPUT_DIR, "adve_results.json")
        )

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    def _print_summary(self, s: dict) -> None:
        verdict = "✅ HYPOTHESIS VALIDATED" if s["hypothesis_validated"] else "⚠️  NEEDS REFINEMENT"
        print(f"\n{'='*55}")
        print(f"  {verdict}")
        print(f"{'='*55}")
        print(f"  Total frames         : {s['total_frames']}")
        print(f"  Encoder calls        : {s['encoder_calls']}")
        print(f"  Delta frames         : {s['delta_frames']}")
        print(f"  Encoder savings      : {s['encoder_savings_pct']}%")
        print(f"  Mean cosine sim (Δ)  : {s['mean_delta_cosine_sim']}")
        print(f"  Min  cosine sim (Δ)  : {s['min_delta_cosine_sim']}")
        print(f"  Frames ≥ threshold   : {s['pct_above_threshold']}%")
        print(f"  Effective FPS        : {s['effective_fps']}")
        print(f"{'='*55}\n")
