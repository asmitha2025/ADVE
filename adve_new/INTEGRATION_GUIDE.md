# Integration Guide — Tiled Encoding + OCR + Unified Search
## How to plug the new modules into the existing ADVE v2.0 server

---

## Step 1: Install dependencies

```bash
pip install easyocr
# easyocr installs: torch-based OCR, supports 80+ languages
# First run downloads model weights (~100MB)

# Whisper already installed from before
# pip install openai-whisper
```

---

## Step 2: File structure to add

```
adve_v2/adve/
├── vision/
│   ├── __init__.py
│   ├── tiled_encoder.py      ← small object search
│   ├── ocr_extractor.py      ← text in video search
│   └── unified_search.py     ← merges all signals
├── audio/
│   ├── __init__.py
│   ├── indexer.py            ← Whisper transcription
│   └── multimodal_search.py  ← merge visual + audio
```

Create the __init__.py:
```bash
touch adve_v2/adve/vision/__init__.py
```

---

## Step 3: Update server.py index_video_task

Replace the existing index_video_task with this:

```python
# In adve_v2/adve/api/server.py

from adve.vision.tiled_encoder  import TiledEncoder
from adve.vision.ocr_extractor  import OCRExtractor
from adve.vision.unified_search import UnifiedSearchEngine
from adve.audio.indexer         import AudioIndexer

# Initialize once at startup (after pipeline init)
tiled_encoder  = TiledEncoder(
    clip_model = anchor_proc.clip_model,
    clip_prep  = anchor_proc.clip_preprocess,
    device     = Config().DEVICE,
)
ocr_extractor = OCRExtractor(
    db_path = "data/main_index/ocr.db",
    device  = Config().DEVICE,
)
audio_indexer = AudioIndexer(
    search_index = search_index,
    device       = Config().DEVICE,
)
unified_search = UnifiedSearchEngine(
    visual_index = search_index,
    ocr_extractor = ocr_extractor,
    audio_indexer = audio_indexer,
)


def index_video_task(video_path: str, video_id: str):
    """Enhanced indexing: visual + tiles + OCR + audio."""
    import cv2

    config   = Config()
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    skip     = max(1, int(round(fps / 5)))  # process at 5 FPS

    frame_idx      = 0
    processed_idx  = 0
    batch          = []
    anchor_timestamps = []

    with pipeline_lock:
        pipeline.reset()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            timestamp = frame_idx / fps

            with pipeline_lock:
                result = pipeline.process_frame(frame, processed_idx)

            # ── Standard global embedding (existing) ──
            batch.append({
                "video_path": video_id,
                "camera_id":  video_id,
                "timestamp":  timestamp,
                "frame_idx":  frame_idx,
                "embedding":  result["embedding"],
                "is_anchor":  result["is_anchor"],
            })

            if result["is_anchor"]:
                anchor_timestamps.append(timestamp)

                # ── Tiled embeddings (NEW: small objects) ──
                tile_results = tiled_encoder.encode_frame(frame, grid="2x2")
                for tile in tile_results[1:]:  # skip global (already added)
                    batch.append({
                        "video_path": video_id,
                        "camera_id":  f"{video_id} [TILE:{tile['tile_id']}]",
                        "timestamp":  timestamp,
                        "frame_idx":  frame_idx,
                        "embedding":  tile["embedding"],
                        "is_anchor":  True,
                    })

            if len(batch) >= 200:
                search_index.add_batch(batch)
                batch = []

            processed_idx += 1

        frame_idx += 1

    cap.release()

    if batch:
        search_index.add_batch(batch)
    search_index.save()

    # ── OCR indexing (NEW: text in video) ──
    print(f"Running OCR on {len(anchor_timestamps)} anchor frames...")
    ocr_extractor.index_video(video_path, video_id, anchor_timestamps)

    # ── Audio indexing (NEW: spoken words) ──
    print(f"Running Whisper transcription...")
    audio_indexer.index_video(video_path, video_id)

    print(f"Indexing complete: {video_id}")
```

---

## Step 4: Update search endpoint

```python
# In server.py, replace search_text endpoint:

@app.post("/v1/search/text")
async def search_text(request: TextSearchRequest):
    """
    Unified search: visual + OCR text + audio.
    """
    results = unified_search.search(
        query    = request.query,
        video_id = request.camera_id or list(_indexed_videos.keys())[-1],
        k        = 5,
        use_visual = True,
        use_ocr    = True,
        use_audio  = True,
    )

    return [
        {
            "video_path":  r.video_id,
            "timestamp":   r.timestamp,
            "similarity":  r.similarity,
            "sources":     r.sources,      # ["visual", "ocr", "audio"]
            "text_found":  r.text_found,   # OCR text at this moment
            "audio_text":  r.audio_text,   # spoken words at this moment
            "frame_idx":   r.frame_idx,
        }
        for r in results
    ]
```

---

## Step 5: Update Gradio demo to show sources

```python
# In demo_v2.py, update the results HTML:

from adve.vision.unified_search import UnifiedSearchEngine

def run_search(query, video_id, ...):
    results = unified_search.search(query, video_id, k=5)

    html_rows = []
    for r in results:
        ts       = fmt_ts(r.timestamp)
        pct      = round(r.similarity * 100, 1)
        badges   = UnifiedSearchEngine.source_badge(r.sources)
        ocr_info = f'<br><small>📝 Text: "{r.text_found}"</small>' if r.text_found else ""
        aud_info = f'<br><small>🎤 Said: "{r.audio_text}"</small>' if r.audio_text else ""

        html_rows.append(f"""
<div class="result-block">
  <span class="ts-badge">⏱ {ts}</span>
  &nbsp;<span class="sim-badge">{pct}% match</span>
  <br>{badges}
  {ocr_info}
  {aud_info}
</div>""")
```

---

## What Each Signal Finds

```
Query: "gradient descent formula"

Visual (CLIP):
  Finds: frames that LOOK LIKE math/ML content
  Misses: exact formula if visually generic

OCR (EasyOCR):
  Finds: frames containing words "gradient", "descent", "∂L/∂w"
  Misses: nothing — if it's readable text, OCR finds it

Audio (Whisper):
  Finds: moments where someone SAYS "gradient descent"
  Misses: silent text or visual-only content

Combined:
  Any frame where the formula appears OR is explained OR is said
  → highest recall of any single system
```

---

## Performance Impact

```
Before (visual only):
  Indexing time (5-min video): ~15 seconds
  Search signals: 1 (CLIP visual)
  Small objects: missed
  Text in video: missed
  Audio: missed

After (visual + tiles + OCR + audio):
  Indexing time: ~3-5 minutes for 5-min video
    CLIP visual:   15 sec (unchanged)
    Tiled CLIP:    45 sec (3x more CLIP calls on anchors only)
    OCR:           60-90 sec (EasyOCR on anchor frames)
    Whisper:       30-60 sec (real-time factor ~0.2x on GPU)
  
  Search signals: 3 (visual + text + audio)
  Small objects: found via tiles
  Text in video: found via OCR
  Audio: found via Whisper

  Search speed: unchanged (<1 second)
  Storage: +20-40% more embeddings in FAISS
```

Indexing is slower. Searching is the same. Finding is much better.
For a product: users care about finding. Indexing happens once in background.
The 3-5 minute indexing is acceptable and can be shown as a progress bar.

---

## Test It

After integration, test with this video:
  youtube.com/watch?v=aircAruvnKk  (3Blue1Brown neural networks)

Queries to test:
  "gradient descent"           → should match visual AND audio AND OCR
  "∂L/∂w"                     → OCR only (specific formula text)
  "the weights update"         → audio only (spoken, not shown)
  "neural network diagram"     → visual only (image, no text)

If all four return results → unified search is working correctly.
