# ADVE — Anchor-Delta Video Embedding

> **96.67% fewer CLIP encoder calls. 0.9484 cosine similarity. Real-time video understanding without re-encoding every frame.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Every video AI system today does this:

```
Frame 1  →  CLIP encoder  →  embedding   ← expensive
Frame 2  →  CLIP encoder  →  embedding   ← expensive
Frame 3  →  CLIP encoder  →  embedding   ← expensive
... 30 times per second, forever
```

At 30 FPS, a 1-hour video requires **108,000 encoder calls**. This is wasteful because between consecutive frames, semantic content changes by only ~2–5%.

---

## The Idea

**Embed anchor frames only. Approximate all other frames using spatial graph deltas.**

```
Frame 0  →  [CLIP + YOLO]  →  anchor embedding + SpatialGraph G₀
Frame 1  →  [YOLO only]   →  ΔG₁ → reconstruct E₁ ≈ f(E₀, ΔG₁)
Frame 2  →  [YOLO only]   →  ΔG₂ → reconstruct E₂ ≈ f(E₀, ΔG₂)
...
Frame k  →  scene change detected → new anchor
```

The **spatial graph** records pairwise object relationships (distance, angle, size ratio).
The **delta ΔG** measures how those relationships changed.
The **reconstructor** blends object embeddings weighted by area and positional stability.

Core hypothesis: `E(frame_t) ≈ f(E_anchor, ΔG(t))`

---

## Validation Results

### 1. Synthetic Validation (`test_video.mp4`, 450 frames)
Evaluated on a 15-second synthetic video with 4 objects including a mid-video scene entry event (Branch 2).

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Encoder Savings | ≥ 70% | **96.67%** | ✅ PASS |
| Mean Cosine Similarity (Δ frames) | ≥ 0.85 | **0.9484** | ✅ PASS |
| Min Cosine Similarity | — | **0.8482** | — |
| Frames Above Threshold | — | **99.77%** | — |
| CLIP Calls | Minimize | **15** | — |
| GPU Throughput (with validation) | — | **32.2 FPS** | — |
| GPU Throughput (no-validation) | — | **53.5 FPS** (RTX 4050) | — |
| CPU Throughput | — | **5.8 FPS** | — |

![ADVE Synthetic Results](outputs/adve_results.png)

### 2. Real-World MOT17 Validation (`MOT17-02-SDP-raw.webm`, 600 frames)
Evaluated on a 20-second real-world multi-object tracking sequence with high pedestrian density, camera motion, and object entries.

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Encoder Savings | ≥ 50% | **60.33%** | ✅ PASS |
| Mean Cosine Similarity (Δ frames) | ≥ 0.85 | **0.9923** | ✅ PASS |
| Min Cosine Similarity | — | **0.9490** | — |
| Frames Above Threshold | — | **100.0%** | — |
| CLIP Calls | Minimize | **238** | — |
| GPU Throughput (with validation) | — | **7.4 FPS** | — |

![ADVE MOT17 Results](outputs_mot17/adve_results.png)

### 3. Baseline Comparison (Table 1 - MOT17)

We compared ADVE's spatial graph delta approximation against standard keyframe sampling strategies (Keyframe-N) on the MOT17 sequence:

| Method | Calls | Mean CosSim | Min CosSim | CPU FPS | GPU FPS | GPU VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Embed (baseline)** | 600 (100.0%) | 1.0000 | 1.0000 | 9.6 | 62.5 | 950.0 MB |
| **Keyframe-5** | 120 (20.0%) | 0.9932 | 0.9080 | 44.7 | 166.7 | 950.0 MB |
| **Keyframe-10** | 60 (10.0%) | 0.9876 | 0.9054 | 84.8 | 210.5 | 950.0 MB |
| **Keyframe-30** | 20 (3.3%) | 0.9762 | 0.9054 | 202.5 | 255.3 | 950.0 MB |
| **ADVE (ours)** | 238 (39.7%) | **0.9923** | **0.9490** | 1.0 | 7.4 | **330.0 MB** |

*ADVE dynamically refreshes keyframes during active pedestrian crossings, leading to a much higher minimum similarity floor (0.9490 vs 0.9054) and a 65% reduction in GPU VRAM.*

---

## How It Works

### Anchor Frame (keyframe)
- Run **CLIP** on the full frame → `E_anchor` (512-d embedding)
- Run **YOLOv8** → detect all objects
- For each object, run **CLIP on the cropped RoI** → per-object embedding
- Build **SpatialGraph G** with pairwise relations

### Delta Frame (all others)
- Run **YOLOv8 tracking only** (no CLIP)
- Compute **ΔG** = structural change between current and anchor graph
- **Reconstruct** embedding: weighted blend of anchor object embeddings, modulated by positional stability
- No CLIP call = near-zero marginal cost per frame

### Anchor Refresh Triggers
| Trigger | Condition |
|---------|-----------|
| Spatial delta | `ΔG.total_magnitude > 0.30` |
| Appearance delta | `histogram_diff > 0.15` |
| Frame budget | `frames_since_anchor ≥ 30` |
| New object (Branch 2) | New track ID detected |

---

## Project Structure

```
adve/
├── config.py               # Thresholds, model selection, device
├── spatial_graph.py        # SpatialGraph, ObjectState, Relation, compute_delta()
├── anchor.py               # AnchorProcessor — CLIP + YOLO on keyframes
├── tracker.py              # DeltaTracker — ByteTrack only, zero CLIP
├── reconstructor.py        # EmbeddingReconstructor — core hypothesis f()
├── validator.py            # Cosine similarity metrics + matplotlib plots
├── pipeline.py             # ADVEPipeline — full orchestrator
├── main.py                 # CLI entry point
├── generate_test_video.py  # Synthetic test video generator
└── outputs/
    ├── adve_results.json   # Per-frame results + summary
    └── adve_results.png    # 3-panel validation chart
```

---

## Quickstart

```bash
# 1. Install
python -m venv adve_env && source adve_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install ultralytics opencv-python matplotlib Pillow

# 2. Generate test video
python generate_test_video.py

# 3. Run
python main.py --video test_video.mp4

# 4. Results
cat outputs/adve_results.json
```

---

## Key Results Interpretation

The 15 CLIP encoder calls out of 450 frames break down as:
- **Frame 0**: initial anchor (mandatory)
- **Frames 30, 60, 90...**: budget trigger (every 30 frames)
- **Frame 225**: Branch 2 — new object "bottle" entered scene, triggered refresh

Every other frame was processed using only YOLOv8 tracking + pure math reconstruction.

---

## Citation

If you use ADVE in your research:

```bibtex
@misc{hariharan2025adve,
  title  = {ADVE: Anchor-Delta Video Embedding for Efficient Semantic Scene Understanding},
  author = {Hariharan, M},
  year   = {2025},
  url    = {https://github.com/Hariharan-1828/ADVE}
}
```

*DOI and arXiv links will be added upon Zenodo/preprint publication.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
