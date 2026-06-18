import torch
from dataclasses import dataclass

@dataclass
class Config:
    # --- Model ---
    CLIP_MODEL: str = "ViT-B/32"
    YOLO_MODEL: str = "yolov8n.pt"

    # --- Anchor Refresh Triggers ---
    SPATIAL_THRESHOLD: float = 0.30       # normalized ΔG magnitude
    APPEARANCE_THRESHOLD: float = 0.15    # histogram correlation drop
    MAX_DELTA_FRAMES: int = 30            # force keyframe every N frames regardless

    # --- Validation ---
    SUCCESS_THRESHOLD: float = 0.85       # min cosine similarity to pass

    # --- Hardware ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- I/O ---
    OUTPUT_DIR: str = "outputs"
