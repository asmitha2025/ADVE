"""
Run this first before fixing anything.
It tells you exactly what is slow.

Usage:
    python profiler.py --video your_video.mp4
"""

import cv2
import time
import numpy as np
import torch
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def profile(video_path: str, max_frames: int = 150):
    print(f"\n{'='*55}")
    print(f"  ADVE Speed Profiler")
    print(f"  Video: {Path(video_path).name}")
    print(f"{'='*55}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("⚠️  GPU NOT DETECTED — this is likely your problem")
        print("   Everything will be 10-50x slower on CPU\n")

    timings = {}
    frame_count = 0

    # ── Load video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(fps // 5))  # process at 5 FPS

    # Read first 5 frames for profiling
    frames = []
    while cap.isOpened() and len(frames) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("Could not read video frames")
        return

    frame = frames[0]
    print(f"Frame size: {frame.shape[1]}×{frame.shape[0]}\n")

    # ── Test 1: YOLO standard ─────────────────────────────────────────────
    print("Testing: YOLO standard (full frame)...")
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)

    # Warmup
    for _ in range(3):
        yolo(frame, verbose=False)

    t = time.time()
    for _ in range(10):
        yolo(frame, verbose=False)
    yolo_ms = (time.time() - t) / 10 * 1000
    timings["YOLO standard (ms/frame)"] = yolo_ms
    print(f"  → {yolo_ms:.1f}ms per frame\n")

    # ── Test 2: CLIP ──────────────────────────────────────────────────────
    print("Testing: CLIP ViT-B/32...")
    import clip
    from PIL import Image

    clip_model, clip_prep = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = clip_prep(pil).unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            clip_model.encode_image(tensor)

    t = time.time()
    with torch.no_grad():
        for _ in range(10):
            clip_model.encode_image(tensor)
    clip_ms = (time.time() - t) / 10 * 1000
    timings["CLIP ViT-B/32 (ms/frame)"] = clip_ms
    print(f"  → {clip_ms:.1f}ms per frame\n")

    # ── Test 3: SAHI slicing ──────────────────────────────────────────────
    print("Testing: SAHI slicing...")
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        sahi_model = AutoDetectionModel.from_pretrained(
            model_type           = "yolov8",
            model_path           = "yolov8n.pt",
            confidence_threshold = 0.2,
            device               = device,
        )

        t = time.time()
        for _ in range(3):
            get_sliced_prediction(
                frame, sahi_model,
                slice_height=320, slice_width=320,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=False,
            )
        sahi_ms = (time.time() - t) / 3 * 1000
        timings["SAHI sliced YOLO (ms/frame)"] = sahi_ms
        print(f"  → {sahi_ms:.1f}ms per frame\n")

    except ImportError:
        print("  SAHI not installed\n")
        sahi_ms = 0

    # ── Test 4: EasyOCR ───────────────────────────────────────────────────
    print("Testing: EasyOCR...")
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)

        t = time.time()
        for _ in range(3):
            reader.readtext(frame)
        ocr_ms = (time.time() - t) / 3 * 1000
        timings["EasyOCR (ms/frame)"] = ocr_ms
        print(f"  → {ocr_ms:.1f}ms per frame\n")

    except ImportError:
        print("  EasyOCR not installed\n")
        ocr_ms = 0

    # ── Test 5: Tiled CLIP (4 tiles) ─────────────────────────────────────
    print("Testing: Tiled CLIP (4 tiles per frame)...")
    h, w = frame.shape[:2]
    tiles = [
        frame[:h//2, :w//2],
        frame[:h//2, w//2:],
        frame[h//2:, :w//2],
        frame[h//2:, w//2:],
    ]

    t = time.time()
    for _ in range(5):
        with torch.no_grad():
            for tile in tiles:
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                t2  = clip_prep(pil).unsqueeze(0).to(device)
                clip_model.encode_image(t2)
    tiled_ms = (time.time() - t) / 5 * 1000
    timings["Tiled CLIP 4x (ms/frame)"] = tiled_ms
    print(f"  → {tiled_ms:.1f}ms per frame\n")

    # ── Test 6: Whisper (on 10-sec audio) ────────────────────────────────
    print("Testing: Whisper base model (10-second audio)...")
    try:
        import whisper, tempfile, subprocess

        wmodel = whisper.load_model("base", device=device)

        # Generate 10 seconds of silence for timing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=16000:cl=mono",
            "-t", "10", tmp_path,
        ], capture_output=True)

        t = time.time()
        wmodel.transcribe(tmp_path, verbose=False)
        whisper_ms = (time.time() - t) * 1000
        timings["Whisper base (ms per 10s audio)"] = whisper_ms
        print(f"  → {whisper_ms:.0f}ms per 10 seconds of audio")
        print(f"     = {whisper_ms/10:.0f}ms per second of video\n")

        import os
        os.unlink(tmp_path)

    except Exception as e:
        print(f"  Whisper test skipped: {e}\n")
        whisper_ms = 0

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PROFILING RESULTS")
    print(f"{'='*55}")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, int(ms / 50))
        print(f"  {name:<40} {ms:>8.1f}ms  {bar}")

    # Project to 1-minute video
    print(f"\n{'─'*55}")
    print(f"  PROJECTED TIME FOR 1-MINUTE VIDEO (at 5 FPS)")
    print(f"{'─'*55}")

    frames_1min = 300  # 1 min × 5 FPS
    anchors     = int(frames_1min * 0.4)  # assume 40% anchor rate on real video
    deltas      = frames_1min - anchors

    ops = {
        "YOLO on all frames":         timings.get("YOLO standard (ms/frame)", 0) * frames_1min / 1000,
        "CLIP on anchor frames only":  timings.get("CLIP ViT-B/32 (ms/frame)", 0) * anchors / 1000,
        "SAHI on anchor frames":       timings.get("SAHI sliced YOLO (ms/frame)", 0) * anchors / 1000,
        "Tiled CLIP on anchors":       timings.get("Tiled CLIP 4x (ms/frame)", 0) * anchors / 1000,
        "OCR on anchor frames":        timings.get("EasyOCR (ms/frame)", 0) * anchors / 1000,
        "Whisper (60s audio)":         timings.get("Whisper base (ms per 10s audio)", 0) * 6 / 1000,
    }

    total = 0
    for name, sec in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {name:<40} {sec:>6.1f}s")
        total += sec

    print(f"{'─'*55}")
    print(f"  TOTAL (all features on):              {total:>6.1f}s = {total/60:.1f} min")
    print(f"\n  YOUR BOTTLENECK IS: {max(ops.items(), key=lambda x: x[1])[0]}")
    print(f"{'='*55}\n")

    return timings


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    args = p.parse_args()
    profile(args.video)
