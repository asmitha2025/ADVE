from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import shutil
import time
import uvicorn

app = FastAPI(
    title="ADVE Video Intelligence API",
    description="Semantic video indexing and search using Anchor-Delta Video Embedding",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
import functools
from adve.search.index import ADVESearchIndex
from adve.core.pipeline import ADVEPipeline
from adve.core.config   import Config
from adve.core.stream   import MultiCameraManager

DATA_DIR = os.environ.get("ADVE_DATA_DIR", "adve_v2/data")
INDEX_DIR = os.path.join(DATA_DIR, "main_index")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

import threading

search_index = ADVESearchIndex(INDEX_DIR)
camera_mgr   = MultiCameraManager(index_writer=search_index)
global_pipeline = None
active_tasks = {}
pipeline_lock = threading.Lock()

# Warm up CLIP and Whisper models on the main thread to prevent thread-safety crashes on Windows
try:
    print("[API Startup] Warming up CLIP text encoder...")
    search_index.search_by_text("warmup", k=1)
    print("[API Startup] CLIP model warmed up successfully.")
except Exception as e:
    print(f"[API Startup] Warning: CLIP warmup failed: {e}")

try:
    print("[API Startup] Warming up Whisper model...")
    from adve.core.audio_transcriber import AudioTranscriber
    transcriber = AudioTranscriber(model_name="tiny")
    transcriber._load_model()
    print("[API Startup] Whisper model warmed up successfully.")
except Exception as e:
    print(f"[API Startup] Warning: Whisper warmup failed: {e}")

try:
    print("[API Startup] Warming up ADVE Pipeline (YOLO & Reconstructor)...")
    import numpy as np
    config = Config()
    global_pipeline = ADVEPipeline(
        config,
        clip_model = search_index._clip_model,
        clip_preprocess = search_index._clip_prep
    )
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    global_pipeline.process_frame(dummy_frame, 0, no_validation=True)
    print("[API Startup] ADVE Pipeline warmed up successfully.")
except Exception as e:
    print(f"[API Startup] Warning: ADVE Pipeline warmup failed: {e}")

global_audio_indexer = None
try:
    print("[API Startup] Initializing and warming up AudioIndexer...")
    from adve.audio.indexer import AudioIndexer
    global_audio_indexer = AudioIndexer(
        search_index,
        device = "cpu",
        clip_model = search_index._clip_model,
        clip_prep = search_index._clip_prep
    )
    global_audio_indexer._get_whisper()
    global_audio_indexer._get_clip()
    print("[API Startup] AudioIndexer warmed up successfully.")
except Exception as e:
    print(f"[API Startup] Warning: AudioIndexer warmup failed: {e}")

global_tiled_encoder = None
global_ocr_extractor = None
global_unified_search = None

try:
    print("[API Startup] Initializing vision and search extensions...")
    from adve.vision.tiled_encoder import TiledEncoder
    from adve.vision.ocr_extractor import OCRExtractor
    from adve.vision.unified_search import UnifiedSearchEngine
    
    config = Config()
    if global_pipeline is not None:
        global_tiled_encoder = TiledEncoder(
            clip_model = global_pipeline.anchor_proc.clip_model,
            clip_prep  = global_pipeline.anchor_proc.clip_preprocess,
            device     = config.DEVICE,
        )
    
    ocr_db_path = os.path.join(INDEX_DIR, "ocr.db")
    os.makedirs(os.path.dirname(ocr_db_path), exist_ok=True)
    global_ocr_extractor = OCRExtractor(
        db_path = ocr_db_path,
        device  = config.DEVICE,
    )
    
    global_unified_search = UnifiedSearchEngine(
        visual_index = search_index,
        ocr_extractor = global_ocr_extractor,
        audio_indexer = global_audio_indexer,
    )
    print("[API Startup] Vision and search extensions initialized successfully.")
except Exception as e:
    print(f"[API Startup] Warning: Failed to initialize extensions: {e}")


def init_users_db():
    import sqlite3
    # Use the same DB as metadata
    db = sqlite3.connect(os.path.join(INDEX_DIR, "metadata.db"))
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT,
            email      TEXT UNIQUE,
            api_key    TEXT UNIQUE,
            created_at REAL
        )
    """)
    db.commit()
    db.close()

# Initialize DB on load
init_users_db()


# ── Models ────────────────────────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query:          str
    k:              int = 10
    camera_id:      Optional[str] = None
    anchor_only:    Optional[bool] = False
    min_similarity: Optional[float] = 0.0
    temporal_dedup: Optional[bool] = True

class StreamRequest(BaseModel):
    camera_id: str
    rtsp_url:  str

class SearchResult(BaseModel):
    video_path: str
    camera_id:  str
    timestamp:  float
    frame_idx:  int
    similarity: float
    is_anchor:  Optional[bool] = False
    source:     Optional[str] = "visual"
    text:       Optional[str] = ""
    objects:    Optional[List[str]] = []

class RegisterRequest(BaseModel):
    name:  str
    email: str

class YouTubeIndexRequest(BaseModel):
    url: str


# ── Background indexing ───────────────────────────────────────────────────────

def ocr_video_async(video_path: str, video_id: str, anchor_timestamps: list):
    """Runs EasyOCR in a background thread."""
    if global_ocr_extractor is not None and anchor_timestamps:
        try:
            print(f"[API Background] Running OCR on {len(anchor_timestamps)} anchor frames...")
            global_ocr_extractor.index_video(video_path, video_id, anchor_timestamps)
            print(f"[API Background] OCR indexing completed for {video_id}.")
        except Exception as e:
            print(f"OCR indexing warning: {e}")

def transcribe_video_async(video_path: str, video_id: str):
    """Runs Whisper transcription and AudioIndexer in a background thread."""
    from adve.core.audio_transcriber import AudioTranscriber
    try:
        transcriber = AudioTranscriber()
        if transcriber.whisper_available:
            print(f"[API Background] Extracting and transcribing speech for {video_id}...")
            speech_segments = transcriber.transcribe(video_path)
            if speech_segments:
                search_index.add_transcripts(video_id, speech_segments)
                print(f"[API Background] Successfully indexed {len(speech_segments)} speech segments.")
    except Exception as e:
        print(f"Audio transcription warning: {e}")

    if global_audio_indexer is not None:
        try:
            print(f"[API Background] Running AudioIndexer on {video_id}...")
            global_audio_indexer.index_video(video_path, video_id)
            print(f"[API Background] AudioIndexer indexing completed for {video_id}.")
        except Exception as e:
            print(f"[API Background] AudioIndexer failed: {e}")

def index_video_task(video_path: str, video_id: str, task_id: Optional[str] = None):
    """Runs in background after upload. Decodes frames, runs ADVE, and schedules OCR/Audio indexing."""
    import cv2
    import numpy as np

    global global_pipeline
    pipeline = global_pipeline
    if pipeline is None:
        print("[API Startup fallback] Initializing new pipeline instance inside task...")
        config   = Config()
        pipeline = ADVEPipeline(config)
        
    with pipeline_lock:
        pipeline.reset()
        
    cap      = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Enforce adaptive frame skip (Adaptive FPS Upgrade)
    config = Config()
    from adve.core.frame_filter import FrameFilter
    motion_filter = FrameFilter(motion_threshold=config.MOTION_THRESHOLD)
    
    last_processed_idx = -999
    
    batch    = []
    idx      = 0
    anchor_timestamps = [] # Track anchor timestamps (Tiled Encoding & OCR)

    print(f"Indexing visual frames: {video_id} (Adaptive FPS)")
    if task_id and task_id in active_tasks:
        active_tasks[task_id]["status"] = "Indexing visual frames..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate motion score to determine next skip size dynamically
        if last_processed_idx == -999:
            has_motion = True
            motion_score = 1.0
        else:
            has_motion, motion_score = motion_filter.has_motion(frame)

        # Dynamic skip size mapping
        if motion_score < 0.003:
            current_skip = int(fps / config.MIN_PROCESS_FPS)
        elif motion_score < 0.01:
            current_skip = int(fps / 2.0)
        elif motion_score < 0.03:
            current_skip = int(fps / config.PROCESS_FPS)
        else:
            current_skip = max(1, int(fps / config.MAX_PROCESS_FPS))

        # Check if we should skip the current frame
        if idx - last_processed_idx < current_skip and (idx - last_processed_idx) < int(fps / config.MIN_PROCESS_FPS):
            if not has_motion or (idx - last_processed_idx) < current_skip:
                idx += 1
                continue

        last_processed_idx = idx

        with pipeline_lock:
            result    = pipeline.process_frame(frame, idx)
        timestamp = idx / fps

        # 1. Add global frame embedding
        batch.append({
            "video_path": video_id,
            "camera_id":  video_id,
            "timestamp":  timestamp,
            "frame_idx":  idx,
            "embedding":  result["embedding"],
            "is_anchor":  result["is_anchor"],
        })

        if result["is_anchor"]:
            anchor_timestamps.append(timestamp)

            # Tiled CLIP Encoding on Anchor Frames (Tiled Encoding / Small Objects)
            if global_tiled_encoder is not None:
                try:
                    tile_results = global_tiled_encoder.encode_frame(frame, grid="2x2")
                    for tile in tile_results[1:]: # skip global (already added)
                        batch.append({
                            "video_path": video_id,
                            "camera_id":  f"{video_id} [TILE:{tile['tile_id']}]",
                            "timestamp":  timestamp,
                            "frame_idx":  idx,
                            "embedding":  tile["embedding"],
                            "is_anchor":  True,
                        })
                except Exception as e:
                    print(f"Tiled encoding warning: {e}")

        # 2. Add object-level crop embeddings (finding small details!)
        for obj in result.get("objects", []):
            batch.append({
                "video_path": video_id,
                "camera_id":  f"{video_id} (Object: {obj['class_name']})",
                "timestamp":  timestamp,
                "frame_idx":  idx,
                "embedding":  obj["embedding"],
                "is_anchor":  result["is_anchor"],
            })

        if len(batch) >= 500:
            search_index.add_batch(batch)
            batch = []

        idx += 1
        if task_id and task_id in active_tasks and idx % 15 == 0:
            pct = min(99.0, (idx / total_frames) * 100)
            active_tasks[task_id]["progress"] = round(pct, 1)

    if batch:
        search_index.add_batch(batch)

    cap.release()
    search_index.save()
    print(f"Indexed {idx} visual frames from {video_id}")

    # Set status to 100% and ready (so the user doesn't wait for background Whisper/OCR)
    if task_id and task_id in active_tasks:
        active_tasks[task_id]["status"] = "Ready to search!"
        active_tasks[task_id]["progress"] = 100.0
        # Auto-remove completed tasks after 15 seconds
        def cleanup():
            time.sleep(15)
            active_tasks.pop(task_id, None)
        threading.Thread(target=cleanup, daemon=True).start()

    # Launch background task for EasyOCR
    threading.Thread(
        target=ocr_video_async,
        args=(video_path, video_id, anchor_timestamps),
        daemon=True
    ).start()

    # Launch background task for Whisper & AudioIndexer
    threading.Thread(
        target=transcribe_video_async,
        args=(video_path, video_id),
        daemon=True
    ).start()


def index_youtube_task(url: str, task_id: str):
    """Downloads a YouTube video and indexes it in the background."""
    import yt_dlp
    
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    def ytdl_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded = d.get('downloaded_bytes') or 0
            pct = (downloaded / total * 100) if total > 0 else 0.0
            if task_id in active_tasks:
                active_tasks[task_id]["status"] = "Downloading from YouTube..."
                active_tasks[task_id]["progress"] = round(pct, 1)
        elif d['status'] == 'finished':
            if task_id in active_tasks:
                active_tasks[task_id]["status"] = "Download finished, starting indexing..."
                active_tasks[task_id]["progress"] = 100.0
                
    opts = {
        "format": "mp4/best",
        "outtmpl": os.path.join(UPLOADS_DIR, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [ytdl_hook]
    }
    
    try:
        print(f"[YouTube] Downloading video from {url}...")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            ext = info.get("ext", "mp4")
            filename = f"{video_id}.{ext}"
            filepath = os.path.join(UPLOADS_DIR, filename)
            
            # Double check path
            if not os.path.exists(filepath):
                for file in os.listdir(UPLOADS_DIR):
                    if file.startswith(video_id):
                        filepath = os.path.join(UPLOADS_DIR, file)
                        filename = file
                        break
            
            if os.path.exists(filepath):
                if task_id in active_tasks:
                    active_tasks[task_id]["name"] = filename
                print(f"[YouTube] Downloaded successfully to {filepath}. Starting indexing...")
                index_video_task(filepath, filename, task_id=task_id)
            else:
                if task_id in active_tasks:
                    active_tasks[task_id]["status"] = "Failed: Downloaded file not found"
                    active_tasks[task_id]["progress"] = 0.0
    except Exception as e:
        print(f"[YouTube] Error downloading or indexing {url}: {e}")
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = f"Failed: {str(e)[:30]}"
            active_tasks[task_id]["progress"] = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/register")
async def register_user(request: RegisterRequest):
    """Registers a new user (First User) and returns a unique live API key."""
    import sqlite3
    import secrets
    
    api_key = f"adve_live_{secrets.token_hex(16)}"
    db = sqlite3.connect(os.path.join(INDEX_DIR, "metadata.db"))
    try:
        db.execute(
            "INSERT INTO users VALUES (NULL, ?, ?, ?, ?)",
            (request.name, request.email, api_key, time.time())
        )
        db.commit()
    except sqlite3.IntegrityError:
        # Email already exists, return the existing user's key
        row = db.execute("SELECT api_key FROM users WHERE email=?", (request.email,)).fetchone()
        if row:
            api_key = row[0]
        else:
            raise HTTPException(status_code=400, detail="Email already registered, registration failed.")
    finally:
        db.close()
        
    return {"status": "success", "api_key": api_key}


@app.post("/v1/index/video")
async def index_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and index a video file. Returns immediately, indexes in background."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    dest = os.path.join(UPLOADS_DIR, file.filename)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task_id = f"upload_{int(time.time())}_{file.filename}"
    active_tasks[task_id] = {
        "name": file.filename,
        "status": "Starting upload...",
        "progress": 0.0
    }

    background_tasks.add_task(index_video_task, dest, file.filename, task_id)

    return {
        "status":   "indexing_started",
        "video_id": file.filename,
        "task_id":  task_id,
        "message":  "Indexing running in background. Use /v1/stats to monitor."
    }


@app.post("/v1/index/youtube")
async def index_youtube(
    request: YouTubeIndexRequest,
    background_tasks: BackgroundTasks
):
    """Index a YouTube video directly from a URL. Downloads and indexes in background."""
    task_id = f"youtube_{int(time.time())}"
    active_tasks[task_id] = {
        "name": request.url.split("watch?v=")[-1][:15],
        "status": "Queued for download...",
        "progress": 0.0
    }
    background_tasks.add_task(index_youtube_task, request.url, task_id)
    return {
        "status":   "indexing_started",
        "task_id":  task_id,
        "message":  "YouTube download and indexing running in background. Use /v1/stats to monitor."
    }


@app.post("/v1/index/stream")
async def add_stream(request: StreamRequest):
    """Connect a live RTSP camera stream."""
    try:
        camera_mgr.add_camera(request.camera_id, request.rtsp_url)
        return {"status": "connected", "camera_id": request.camera_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/search/text", response_model=List[SearchResult])
async def search_text(request: TextSearchRequest):
    """Search video content using a natural language query."""
    query_k = max(request.k * 20, 500) if request.anchor_only else max(request.k * 5, 100)
    raw_results = search_index.search_by_text(request.query, k=query_k)
    
    scene_matches = []
    object_matches = []
    for r in raw_results:
        if "[AUDIO]" in r.camera_id or "Speech:" in r.camera_id:
            continue
        elif "(Object:" in r.camera_id:
            object_matches.append(r)
        else:
            scene_matches.append(r)

    # Group matches by (video_path, frame_idx)
    from adve.search.index import normalize_video_path
    frames_dict = {}
    
    for r in scene_matches:
        key = (normalize_video_path(r.video_path), r.frame_idx)
        if key not in frames_dict:
            frames_dict[key] = {
                "video_path": r.video_path,
                "camera_id":  r.camera_id,
                "timestamp":  r.timestamp,
                "frame_idx":  r.frame_idx,
                "similarity": r.similarity,
                "is_anchor":  r.is_anchor,
                "object_matches": [],
                "source_has_scene": True
            }
        else:
            if r.similarity > frames_dict[key]["similarity"]:
                frames_dict[key]["similarity"] = r.similarity
                frames_dict[key]["is_anchor"] = r.is_anchor

    for r in object_matches:
        key = (normalize_video_path(r.video_path), r.frame_idx)
        if key not in frames_dict:
            frames_dict[key] = {
                "video_path": r.video_path,
                "camera_id":  r.video_path,
                "timestamp":  r.timestamp,
                "frame_idx":  r.frame_idx,
                "similarity": 0.15,  # Baseline low similarity
                "is_anchor":  r.is_anchor,
                "object_matches": [],
                "source_has_scene": False
            }
        frames_dict[key]["object_matches"].append(r)

    # Process visual results with boost and objects tagging
    visual_results = []
    
    class ProcessedVisualResult:
        def __init__(self, video_path, camera_id, timestamp, frame_idx, similarity, is_anchor, objects):
            self.video_path = video_path
            self.camera_id = camera_id
            self.timestamp = timestamp
            self.frame_idx = frame_idx
            self.similarity = similarity
            self.is_anchor = is_anchor
            self.objects = objects

    for key, info in frames_dict.items():
        if info["source_has_scene"]:
            raw_boost = max([obj.similarity for obj in info["object_matches"]]) if info["object_matches"] else 0.0
            boost = max(0.0, raw_boost - 0.24)
            final_sim = min(1.0, info["similarity"] + boost * 0.8)
        else:
            raw_boost = max([obj.similarity for obj in info["object_matches"]]) if info["object_matches"] else 0.0
            final_sim = max(0.0, raw_boost - 0.06) if raw_boost > 0.24 else 0.12
        
        # Query database for all objects at this frame to build tags list
        cursor = search_index.db.execute(
            "SELECT camera_id FROM embeddings WHERE video_path = ? AND frame_idx = ? AND camera_id LIKE '%(Object:%'",
            (info["video_path"], info["frame_idx"])
        )
        object_labels = []
        for row in cursor.fetchall():
            cam_id = row[0]
            try:
                label = cam_id.split("(Object: ")[1].rstrip(")")
                object_labels.append(label)
            except Exception:
                pass
        
        visual_results.append(
            ProcessedVisualResult(
                video_path = info["video_path"],
                camera_id  = info["camera_id"],
                timestamp  = info["timestamp"],
                frame_idx  = info["frame_idx"],
                similarity = final_sim,
                is_anchor  = info["is_anchor"],
                objects    = list(set(object_labels))
            )
        )
    
    # 1. Filter by camera_id
    if request.camera_id:
        norm_filter = normalize_video_path(request.camera_id)
        filtered = []
        for r in visual_results:
            r_norm_path = normalize_video_path(r.video_path)
            r_norm_cam = normalize_video_path(r.camera_id) if '/' in r.camera_id or '\\' in r.camera_id or '.' in r.camera_id else None
            
            if (r.camera_id == request.camera_id or 
                r_norm_path == norm_filter or 
                (r_norm_cam is not None and r_norm_cam == norm_filter) or
                os.path.basename(r.video_path) == request.camera_id or
                os.path.basename(r.video_path) == os.path.basename(request.camera_id)):
                filtered.append(r)
        visual_results = filtered
        
    # 2. Filter by anchor_only
    if request.anchor_only:
        visual_results = [r for r in visual_results if r.is_anchor]

    # Query audio results using AudioIndexer
    audio_results = []
    if global_audio_indexer is not None:
        if request.camera_id:
            # Query only the requested video
            audio_results = global_audio_indexer.search(request.query, video_id=request.camera_id, k=query_k)
        else:
            # Query all indexed videos
            videos = [v["video_path"] for v in search_index.stats()["indexed_videos"]]
            unique_videos = list(set(videos))
            for v_id in unique_videos:
                audio_results.extend(global_audio_indexer.search(request.query, video_id=v_id, k=query_k))

    # Merge visual and audio results
    min_gap = 8.0 if request.temporal_dedup else 0.0
    from adve.audio.multimodal_search import merge_results
    merged = merge_results(visual_results, audio_results, min_gap=min_gap, top_k=request.k)

    # 3. Filter by similarity threshold
    if request.min_similarity > 0.0:
        merged = [r for r in merged if r.similarity >= request.min_similarity]

    results = []
    for r in merged:
        if r.source == "audio":
            camera_id = f"{r.video_path} (Speech: \"{r.text}\")"
        elif r.source == "both":
            camera_id = f"{r.video_path} (Speech: \"{r.text}\")"
        else:
            camera_id = getattr(r, "camera_id", r.video_path)

        frame_idx = getattr(r, "frame_idx", 0)
        if frame_idx == 0 and r.timestamp > 0:
            frame_idx = int(r.timestamp * 30)

        results.append(
            SearchResult(
                video_path = r.video_path,
                camera_id  = camera_id,
                timestamp  = r.timestamp,
                frame_idx  = frame_idx,
                similarity = r.similarity,
                is_anchor  = r.is_anchor,
                source     = r.source,
                text       = r.text,
                objects    = getattr(r, "objects", []),
            )
        )

    # ── Confidence gate ─────────────────────────────────────────────────────
    # Reject results whose best similarity is below a hard floor so callers
    # never receive confidently-wrong answers for absent queries.
    WEAK_THRESHOLD = 0.22
    if results and results[0].similarity < WEAK_THRESHOLD:
        return []

    return results[:request.k]


@app.post("/v1/search/image", response_model=List[SearchResult])
async def search_image(
    file: UploadFile = File(...),
    k: int = 10,
    camera_id: Optional[str] = None,
    anchor_only: Optional[bool] = False,
    min_similarity: Optional[float] = 0.0,
    temporal_dedup: Optional[bool] = True
):
    """Search for similar scenes using an image."""
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    query_k = max(k * 20, 500) if anchor_only else max(k * 5, 100)
    results = search_index.search_by_image(img, k=query_k)
    
    # 1. Filter by camera_id
    if camera_id:
        from adve.search.index import normalize_video_path
        norm_filter = normalize_video_path(camera_id)
        filtered = []
        for r in results:
            r_norm_path = normalize_video_path(r.video_path)
            r_norm_cam = normalize_video_path(r.camera_id) if '/' in r.camera_id or '\\' in r.camera_id or '.' in r.camera_id else None
            
            if (r.camera_id == camera_id or 
                r_norm_path == norm_filter or 
                (r_norm_cam is not None and r_norm_cam == norm_filter) or
                os.path.basename(r.video_path) == camera_id or
                os.path.basename(r.video_path) == os.path.basename(camera_id)):
                filtered.append(r)
        results = filtered

    # 2. Filter by anchor_only
    if anchor_only:
        results = [r for r in results if r.is_anchor]

    # 3. Filter by similarity threshold
    if min_similarity > 0.0:
        results = [r for r in results if r.similarity >= min_similarity]

    # 4. Temporal deduplication
    if temporal_dedup:
        deduped = []
        for r in results:
            if not any(
                r.video_path == accepted.video_path and abs(r.timestamp - accepted.timestamp) < 8.0 
                for accepted in deduped
            ):
                deduped.append(r)
        results = deduped

    return [
        SearchResult(
            video_path = r.video_path,
            camera_id  = r.camera_id,
            timestamp  = r.timestamp,
            frame_idx  = r.frame_idx,
            similarity = r.similarity,
            is_anchor  = r.is_anchor,
        )
        for r in results[:k]
    ]


@app.get("/v1/stats")
async def get_stats():
    return {
        "index":   search_index.stats(),
        "cameras": camera_mgr.status(),
        "active_tasks": active_tasks
    }


@app.get("/", response_class=HTMLResponse)
async def get_landing():
    html_path = os.path.join(os.path.dirname(__file__), "landing.html")
    if not os.path.exists(html_path):
        return "<h3>ADVE Product Landing Page (landing.html) not found!</h3>"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(html_path):
        return "<h3>ADVE Dashboard (dashboard.html) not found!</h3>"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@functools.lru_cache(maxsize=256)
def read_frame_cached(video_path: str, frame_idx: int) -> Optional[bytes]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


@app.get("/v1/frame")
async def get_frame(video_id: str, frame_idx: int):
    # Try absolute or relative path first
    video_path = video_id if os.path.exists(video_id) else os.path.join(UPLOADS_DIR, video_id)
    
    if not os.path.exists(video_path):
        if "MOT17" in video_id:
            video_path = f"Input video/{video_id}"
        elif video_id == "test_video.mp4":
            video_path = "test_video.mp4"
        else:
            # Check candidate directories
            for folder in ["demo_videos", "demo_data/videos", "adve_v2/demo_videos", "adve_v2/demo_data/videos"]:
                candidate = os.path.join(folder, os.path.basename(video_id))
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            else:
                video_path = video_id
            
    if not os.path.exists(video_path):
        basename = os.path.basename(video_id)
        fallback = os.path.join(UPLOADS_DIR, basename)
        if os.path.exists(fallback):
            video_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_id}")
            
    buffer_bytes = read_frame_cached(video_path, frame_idx)
    if buffer_bytes is None:
        raise HTTPException(status_code=400, detail=f"Could not read frame {frame_idx}")
        
    return Response(content=buffer_bytes, media_type="image/jpeg")


@app.get("/v1/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
