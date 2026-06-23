import torch
from dataclasses import dataclass

@dataclass
class Config:
    # --- Model ---
    CLIP_MODEL: str = "ViT-L/14@336px"
    YOLO_MODEL: str = "yolov8m.pt"

    # --- Anchor Refresh Triggers ---
    SPATIAL_THRESHOLD: float = 0.30       # normalized ΔG magnitude
    APPEARANCE_THRESHOLD: float = 0.15    # histogram correlation drop
    MAX_DELTA_FRAMES: int = 30            # force keyframe every N frames regardless

    # --- Validation ---
    SUCCESS_THRESHOLD: float = 0.85       # min cosine similarity to pass

    # --- Hardware ---
    # --- Hardware ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_DEVICE: str = "cpu"  # Keep CLIP on CPU to prevent VRAM Out-of-Memory (OOM) crashes
    YOLO_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_IMGSZ: int = 320  # Optimized input resolution
    
    # --- Performance Tuning ---
    PROCESS_FPS: int = 5          # Target FPS for indexing (downsampling from native FPS)
    MIN_PROCESS_FPS: float = 0.5   # Downsample to 0.5 FPS (1 frame every 2 seconds) in static scenes
    MAX_PROCESS_FPS: float = 15.0  # Up to 15 FPS for high-motion action scenes
    MOTION_THRESHOLD: float = 0.02   # Skip YOLO if motion score is below this threshold
    YOLO_HALF: bool = True        # FP16 YOLO (30% faster on GPU)

    # --- I/O ---
    OUTPUT_DIR: str = "outputs"
    MLP_MODEL_PATH: str = "training/checkpoints/best_model.pt"
