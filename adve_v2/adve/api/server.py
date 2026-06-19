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
from adve.search.index import ADVESearchIndex
from adve.core.pipeline import ADVEPipeline
from adve.core.config   import Config
from adve.core.stream   import MultiCameraManager

search_index = ADVESearchIndex("data/main_index")
camera_mgr   = MultiCameraManager(index_writer=search_index)


def init_users_db():
    import sqlite3
    # Use the same DB as metadata
    db = sqlite3.connect("data/main_index/metadata.db")
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
    query:     str
    k:         int = 10
    camera_id: Optional[str] = None

class StreamRequest(BaseModel):
    camera_id: str
    rtsp_url:  str

class SearchResult(BaseModel):
    video_path: str
    camera_id:  str
    timestamp:  float
    frame_idx:  int
    similarity: float

class RegisterRequest(BaseModel):
    name:  str
    email: str


# ── Background indexing ───────────────────────────────────────────────────────

def index_video_task(video_path: str, video_id: str):
    """Runs in background after upload. Decodes frames, runs ADVE, and transcribes audio."""
    import cv2
    import numpy as np
    from adve.core.audio_transcriber import AudioTranscriber

    config   = Config()
    pipeline = ADVEPipeline(config)
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    batch    = []
    idx      = 0

    print(f"Indexing: {video_id}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result    = pipeline.process_frame(frame, idx)
        timestamp = idx / fps

        batch.append({
            "video_path": video_id,
            "camera_id":  video_id,
            "timestamp":  timestamp,
            "frame_idx":  idx,
            "embedding":  result["embedding"],
            "is_anchor":  result["is_anchor"],
        })

        if len(batch) >= 500:
            search_index.add_batch(batch)
            batch = []

        idx += 1

    if batch:
        search_index.add_batch(batch)

    cap.release()
    search_index.save()
    print(f"Indexed {idx} visual frames from {video_id}")

    # Speech transcription using Whisper
    try:
        transcriber = AudioTranscriber()
        if transcriber.whisper_available:
            print(f"Extracting and transcribing speech for {video_id}...")
            speech_segments = transcriber.transcribe(video_path)
            if speech_segments:
                search_index.add_transcripts(video_id, speech_segments)
                print(f"Successfully indexed {len(speech_segments)} speech segments.")
    except Exception as e:
        print(f"Audio transcription warning: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/register")
async def register_user(request: RegisterRequest):
    """Registers a new user (First User) and returns a unique live API key."""
    import sqlite3
    import secrets
    
    api_key = f"adve_live_{secrets.token_hex(16)}"
    db = sqlite3.connect("data/main_index/metadata.db")
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
    os.makedirs("data/uploads", exist_ok=True)
    dest = f"data/uploads/{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(index_video_task, dest, file.filename)

    return {
        "status":   "indexing_started",
        "video_id": file.filename,
        "message":  "Indexing running in background. Use /v1/stats to monitor."
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
    results = search_index.search_by_text(request.query, k=request.k)
    return [
        SearchResult(
            video_path = r.video_path,
            camera_id  = r.camera_id,
            timestamp  = r.timestamp,
            frame_idx  = r.frame_idx,
            similarity = r.similarity,
        )
        for r in results
    ]


@app.post("/v1/search/image", response_model=List[SearchResult])
async def search_image(file: UploadFile = File(...), k: int = 10):
    """Search for similar scenes using an image."""
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = search_index.search_by_image(img, k=k)
    return [
        SearchResult(
            video_path = r.video_path,
            camera_id  = r.camera_id,
            timestamp  = r.timestamp,
            frame_idx  = r.frame_idx,
            similarity = r.similarity,
        )
        for r in results
    ]


@app.get("/v1/stats")
async def get_stats():
    return {
        "index":   search_index.stats(),
        "cameras": camera_mgr.status(),
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


@app.get("/v1/frame")
async def get_frame(video_id: str, frame_idx: int):
    import cv2
    
    video_path = f"data/uploads/{video_id}"
    if not os.path.exists(video_path):
        if "MOT17" in video_id:
            video_path = f"Input video/{video_id}"
        elif video_id == "test_video.mp4":
            video_path = "test_video.mp4"
        else:
            video_path = video_id
            
    if not os.path.exists(video_path):
        basename = os.path.basename(video_id)
        fallback = f"data/uploads/{basename}"
        if os.path.exists(fallback):
            video_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_id}")
            
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=400, detail=f"Could not read frame {frame_idx}")
        
    _, buffer = cv2.imencode(".jpg", frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/v1/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
