from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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


# ── Background indexing ───────────────────────────────────────────────────────

def index_video_task(video_path: str, video_id: str):
    """Runs in background after upload."""
    import cv2
    import numpy as np

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
    print(f"Indexed {idx} frames from {video_id}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

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


@app.get("/v1/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


def run():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
