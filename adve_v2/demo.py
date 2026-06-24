import os
import sys
import time
import base64
import subprocess
import shutil
import cv2
import numpy as np
import gradio as gr
import yt_dlp
import groq

from typing import Optional
import static_ffmpeg
static_ffmpeg.add_paths()


# Ensure adve_v2 is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config
from adve.search.index import ADVESearchIndex, SearchResult, normalize_video_path
from adve.core.audio_transcriber import AudioTranscriber

# Global state to keep track of active index, video, and search results
active_video_path = None
active_search_results = []
index_dir = os.path.join(current_dir, "data", "demo_index")
os.makedirs(index_dir, exist_ok=True)
search_index = ADVESearchIndex(index_dir)

# Warm up CLIP and Whisper models on the main thread to prevent thread-safety crashes on Windows
try:
    print("[Demo Startup] Warming up CLIP text encoder...")
    search_index.search_by_text("warmup", k=1)
    print("[Demo Startup] CLIP model warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: CLIP warmup failed: {e}")

try:
    print("[Demo Startup] Warming up Whisper model...")
    transcriber = AudioTranscriber(model_name="tiny")
    transcriber._load_model()
    print("[Demo Startup] Whisper model warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: Whisper warmup failed: {e}")

try:
    print("[Demo Startup] Warming up ADVE Pipeline (YOLO & Reconstructor)...")
    config = Config()
    warmup_pipeline = ADVEPipeline(
        config,
        clip_model = search_index._clip_model,
        clip_preprocess = search_index._clip_prep
    )
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    warmup_pipeline.process_frame(dummy_frame, 0, no_validation=True)
    print("[Demo Startup] ADVE Pipeline warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: ADVE Pipeline warmup failed: {e}")

global_audio_indexer = None
try:
    print("[Demo Startup] Initializing and warming up AudioIndexer...")
    from adve.audio.indexer import AudioIndexer
    global_audio_indexer = AudioIndexer(
        search_index,
        device = "cpu",
        clip_model = search_index._clip_model,
        clip_prep = search_index._clip_prep
    )
    global_audio_indexer._get_whisper()
    global_audio_indexer._get_clip()
    print("[Demo Startup] AudioIndexer warmed up successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: AudioIndexer warmup failed: {e}")

global_tiled_encoder = None
global_ocr_extractor = None
global_unified_search = None

try:
    print("[Demo Startup] Initializing vision and search extensions...")
    from adve.vision.tiled_encoder import TiledEncoder
    from adve.vision.ocr_extractor import OCRExtractor
    from adve.vision.unified_search import UnifiedSearchEngine
    
    config = Config()
    if warmup_pipeline is not None:
        global_tiled_encoder = TiledEncoder(
            clip_model = warmup_pipeline.anchor_proc.clip_model,
            clip_prep  = warmup_pipeline.anchor_proc.clip_preprocess,
            device     = config.DEVICE,
        )
    
    ocr_db_path = os.path.join(index_dir, "ocr.db")
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
    print("[Demo Startup] Vision and search extensions initialized successfully.")
except Exception as e:
    print(f"[Demo Startup] Warning: Failed to initialize extensions: {e}")




def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None





def download_youtube(url: str, progress=gr.Progress()) -> str:
    """Download the first 5 minutes of a YouTube video, or fall back to low-res full download if ffmpeg is missing."""
    progress(0.05, desc="Checking YouTube URL...")
    os.makedirs("demo_videos", exist_ok=True)

    ffmpeg_ok = is_ffmpeg_available()

    # --- Real-time download progress hook ---
    _last_pct = [0.0]
    def _progress_hook(d):
        if d.get("status") == "downloading":
            total   = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = 0.1 + 0.75 * (downloaded / total)   # maps 0→100% download into 10%→85% of progress bar
                pct = min(0.85, pct)
                if pct - _last_pct[0] >= 0.02:             # only update every 2% to reduce Gradio spam
                    _last_pct[0] = pct
                    mb_done = downloaded / 1_048_576
                    mb_total = total / 1_048_576
                    progress(pct, desc=f"Downloading: {mb_done:.1f} / {mb_total:.1f} MB ({pct*100:.0f}%)")
        elif d.get("status") == "finished":
            progress(0.88, desc="Download complete — preparing video...")

    if ffmpeg_ok:
        opts = {
            "format": "mp4/best",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": 300}],
            "force_keyframes_at_cuts": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_progress_hook],
        }
        progress(0.10, desc="Starting download (first 5 minutes)...")
    else:
        opts = {
            "format": "worst[ext=mp4]/mp4",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_progress_hook],
        }
        progress(0.10, desc="ffmpeg not found — downloading full video in low-res...")

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            ext = info.get("ext", "mp4")
            expected_path = f"demo_videos/{video_id}.{ext}"

            if os.path.exists(expected_path):
                progress(0.90, desc="Video ready for indexing!")
                return expected_path

            # Scan directory as fallback
            for file in os.listdir("demo_videos"):
                if file.startswith(video_id):
                    progress(0.90, desc="Video ready for indexing!")
                    return os.path.join("demo_videos", file)
            raise FileNotFoundError("Downloaded video file not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {e}")


def index_video(video_path: str, sampling_rate: float = 5.0, use_adaptive_fps: bool = True, index_audio: bool = False, index_ocr: bool = False, progress=gr.Progress()) -> str:
    """Run ADVE pipeline to index anchor frames in the video."""
    global active_video_path
    global search_index
    active_video_path = video_path

    progress(0.0, desc="Initializing ADVE pipeline...")
    config = Config()
    # CLIP runs on CPU (for memory stability) while YOLO runs on GPU (for speed) if CUDA is available
    pipeline = ADVEPipeline(
        config,
        clip_model = search_index._clip_model,
        clip_preprocess = search_index._clip_prep
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Calculate step size based on sampling rate (e.g. 1 FPS means processing 1 frame every 'fps' frames)
    frame_step = max(1, int(fps / sampling_rate))
    print(f"[Demo Indexer] Total frames to process: {total_frames} @ {fps:.1f} FPS")
    if use_adaptive_fps:
        print(f"[Demo Indexer] Adaptive FPS Enabled (Base target: {sampling_rate} FPS, boundaries: {config.MIN_PROCESS_FPS}-{config.MAX_PROCESS_FPS} FPS)")
    else:
        print(f"[Demo Indexer] Sampling rate: {sampling_rate} FPS (processing 1 frame every {frame_step} frames)")

    # Clear out any previous database entries to keep the demo clean
    try:
        search_index.clear()
    except Exception as e:
        print(f"Warning: Failed to clear search index cleanly: {e}")
        # Fallback to recreate
        try:
            search_index = ADVESearchIndex(index_dir)
            search_index.clear()
        except Exception as e2:
            print(f"Warning: Recreate fallback failed: {e2}")

    # Set up Adaptive FPS FrameFilter if enabled
    from adve.core.frame_filter import FrameFilter
    motion_filter = FrameFilter(motion_threshold=config.MOTION_THRESHOLD)
    last_processed_idx = -999
    
    cap = cv2.VideoCapture(video_path)
    idx = 0
    sampled_idx = 0
    anchors_count = 0
    start_time = time.time()
    anchor_timestamps = [] # Track anchor timestamps (Tiled Encoding & OCR)
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        should_process = False
        if use_adaptive_fps:
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
                current_skip = int(fps / sampling_rate)
            else:
                current_skip = max(1, int(fps / config.MAX_PROCESS_FPS))

            # Check if we should skip the current frame
            if idx - last_processed_idx < current_skip and (idx - last_processed_idx) < int(fps / config.MIN_PROCESS_FPS):
                if not has_motion or (idx - last_processed_idx) < current_skip:
                    idx += 1
                    continue
            should_process = True
        else:
            if idx % frame_step == 0:
                should_process = True
                
        if should_process:
            last_processed_idx = idx
            result = pipeline.process_frame(frame, idx, no_validation=True)
            
            # Extract detected object labels for Two-Stage search metadata
            obj_classes = [obj["class_name"] for obj in result.get("objects", [])]
            obj_metadata = ", ".join(set(obj_classes))
            
            # Save all processed frames (both anchors and reconstructed deltas) to the search index
            timestamp = idx / fps
            search_index.add(
                video_path,
                "youtube_cam",
                timestamp,
                idx,
                result["embedding"],
                is_anchor=result["is_anchor"],
                text=obj_metadata
            )
            
            if result["is_anchor"]:
                anchors_count += 1
                anchor_timestamps.append(timestamp)
                
                # Tiled CLIP Encoding on Anchor Frames (Tiled Encoding / Small Objects)
                if global_tiled_encoder is not None:
                    try:
                        tile_results = global_tiled_encoder.encode_frame(frame, grid="2x2")
                        for tile in tile_results[1:]: # skip global (already added)
                            search_index.add(
                                video_path,
                                f"youtube_cam [TILE:{tile['tile_id']}]",
                                timestamp,
                                idx,
                                tile["embedding"],
                                is_anchor=True
                            )
                    except Exception as e:
                        print(f"Tiled encoding warning in demo: {e}")
            sampled_idx += 1
            
        idx += 1
        if idx % 15 == 0:
            progress_pct = min(0.99, idx / max(1, total_frames))
            progress(progress_pct, desc=f"Ingested {idx}/{total_frames} frames (Processed {sampled_idx} sampled frames)")

    cap.release()
    search_index.save()

    # Run EasyOCR indexing (Text in Video) - Optional
    if index_ocr and global_ocr_extractor is not None and anchor_timestamps:
        progress(0.85, desc="Running OCR text extraction on anchor frames...")
        try:
            print(f"[Demo Indexer] Running OCR extraction on {len(anchor_timestamps)} anchor frames...")
            video_id = os.path.basename(video_path)
            global_ocr_extractor.index_video(video_path, video_id, anchor_timestamps)
        except Exception as e:
            print(f"[Demo Indexer] OCR extraction failed: {e}")
    
    # Extract and transcribe audio using Whisper - Optional
    if index_audio:
        progress(0.9, desc="Transcribing audio with Whisper...")
        try:
            print(f"[Demo Indexer] Extracting and transcribing audio from {video_path}...")
            transcriber = AudioTranscriber(model_name="tiny")
            segments = transcriber.transcribe(video_path)
            if segments:
                search_index.add_transcripts(video_path, segments)
                print(f"[Demo Indexer] Successfully indexed {len(segments)} audio segments.")
            else:
                print("[Demo Indexer] No audio segments transcribed.")
        except Exception as e:
            print(f"Warning: Audio transcription failed or skipped: {e}")

        # New audio indexing using AudioIndexer
        if global_audio_indexer is not None:
            try:
                print(f"[Demo Indexer] Running AudioIndexer on {video_path}...")
                global_audio_indexer.index_video(video_path, os.path.basename(video_path))
            except Exception as e:
                print(f"[Demo Indexer] AudioIndexer failed: {e}")

    elapsed = time.time() - start_time
    
    savings = 100.0 * (1.0 - (anchors_count / max(1, sampled_idx)))
    summary = (
        f"✅ Indexing Complete!\n"
        f"- Total Video Frames: {idx}\n"
        f"- Sampled Frames Processed: {sampled_idx} (sampling mode: {'Adaptive FPS' if use_adaptive_fps else f'{sampling_rate} FPS'})\n"
        f"- RAG Anchor Chunks Created: {anchors_count}\n"
        f"- ADVE Neural Cost Savings: {savings:.1f}%\n"
        f"- Processing Time: {elapsed:.2f} seconds ({sampled_idx/elapsed:.1f} FPS)"
    )
    return summary


def handle_youtube_index(url: str, sampling_rate: float, use_adaptive_fps: bool, index_audio: bool, index_ocr: bool, progress=gr.Progress()) -> str:
    if not url.strip():
        return "Please enter a valid YouTube URL."
    try:
        video_path = download_youtube(url, progress)
        return index_video(video_path, sampling_rate, use_adaptive_fps, index_audio, index_ocr, progress)
    except Exception as e:
        return f"Error: {e}"


def handle_local_index(file, sampling_rate: float, use_adaptive_fps: bool, index_audio: bool, index_ocr: bool, progress=gr.Progress()) -> str:
    if file is None:
        return "Please upload a video file first."
    return index_video(file, sampling_rate, use_adaptive_fps, index_audio, index_ocr, progress)


def extract_clip(video_path: str, timestamp: float, duration: float = 10.0) -> str:
    """Extract a video clip around the timestamp. Falls back to OpenCV if ffmpeg is missing."""
    os.makedirs("clips", exist_ok=True)
    start = max(0.0, timestamp - 2.0)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join("clips", f"clip_{video_stem}_{timestamp:.1f}_{duration:.1f}.mp4")
    
    # Return cached clip if it exists and is valid
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[Clip Cache] Reusing cached clip: {output_path}")
        return output_path
    
    # Check if ffmpeg is available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        # Try browser-friendly h264 re-encoding first
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except Exception:
            # Fallback to copy mode
            cmd_copy = [
                ffmpeg_bin, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            try:
                subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                return output_path
            except Exception as e:
                print(f"[FFmpeg] Clip extraction failed: {e}")
    else:
        print("[Demo Indexer] ffmpeg not found. Using OpenCV as fallback for clip extraction.")
        
    # OpenCV Fallback (silent video, but visual is extracted)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = max(0, int(start * fps))
        end_frame = min(total_frames, int((start + duration) * fps))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        success = False
        for codec in ['H264', 'X264', 'mp4v']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    success = True
                    break
            except Exception:
                continue
                
        cap.release()
        if success:
            return output_path
    except Exception as e:
        print(f"[OpenCV Fallback] Clip extraction failed: {e}")
        
    return None


def get_dynamic_duration(video_path: str, start_time: float, default_duration: float = 10.0) -> float:
    """Query database for the next anchor frame to calculate the dynamic scene duration."""
    try:
        cursor = search_index.db.execute(
            "SELECT timestamp FROM embeddings WHERE video_path = ? AND timestamp > ? AND is_anchor = 1 ORDER BY timestamp ASC LIMIT 1",
            (video_path, start_time)
        )
        row = cursor.fetchone()
        if row:
            next_ts = row[0]
            # Since clip extraction starts at start_time - 2.0 (see extract_clip):
            # start = max(0.0, start_time - 2.0)
            # The duration should cover from start until the next anchor timestamp + a 1.0 second buffer
            start = max(0.0, start_time - 2.0)
            duration = (next_ts - start) + 1.0
            return max(3.0, min(60.0, duration))
        else:
            # Last scene in video: extract to the end of the video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps
            cap.release()
            start = max(0.0, start_time - 2.0)
            duration = total_duration - start
            return max(3.0, min(60.0, duration))
    except Exception as e:
        print(f"[Dynamic Duration] Error querying database: {e}")
        return default_duration


def search_and_retrieve(query: str, clip_duration: float, use_dynamic_duration: bool, anchor_only: bool = False, min_similarity: float = 0.0):
    """Search natural language and return matching frame images and video clips."""
    global active_search_results, active_video_path
    if not active_video_path:
        return (
            "No video has been indexed yet. Please index a video first.", 
            [], 
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False)
        )
        
    if not query.strip():
        return (
            "Please enter a search query.", 
            [], 
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False)
        )
        
    print(f"[Demo Search] Querying: '{query}'")

    results = []

    if global_unified_search is not None:
        video_id = os.path.basename(active_video_path)
        try:
            # Search using Unified Search Engine (Visual + OCR + Audio)
            unified_results = global_unified_search.search(
                query      = query,
                video_id   = video_id,
                k          = 5,
                use_visual = True,
                use_ocr    = True,
                use_audio  = True,
            )
            
            # Filter by similarity threshold
            if min_similarity > 0.0:
                unified_results = [r for r in unified_results if r.similarity >= min_similarity]

            for r in unified_results:
                text_context = ""
                if r.text_found:
                    text_context = f"Text: \"{r.text_found}\""
                if r.audio_text:
                    if text_context:
                        text_context += " | "
                    text_context += f"Speech: \"{r.audio_text}\""

                camera_id = r.video_id
                if "tile_" in r.tile_id and r.tile_id != "global":
                    camera_id = f"{r.video_id} [TILE:{r.tile_id}]"
                elif r.text_found and not r.audio_text:
                    camera_id = f"{r.video_id} (Text: \"{r.text_found[:30]}...\")"
                elif r.audio_text:
                    camera_id = f"{r.video_id} (Speech: \"{r.audio_text[:30]}...\")"

                frame_idx = r.frame_idx if r.frame_idx > 0 else int(r.timestamp * 30)

                source_str = "visual"
                if "ocr" in r.sources and "audio" in r.sources:
                    source_str = "both"
                elif "audio" in r.sources:
                    source_str = "audio"
                elif "ocr" in r.sources:
                    source_str = "ocr"

                mock_r = SearchResult(
                    video_path = active_video_path,
                    camera_id  = camera_id,
                    timestamp  = r.timestamp,
                    frame_idx  = frame_idx,
                    similarity = r.similarity,
                    is_anchor  = r.is_anchor,
                )
                mock_r.source = source_str
                mock_r.text_found = r.text_found
                mock_r.audio_text = r.audio_text
                mock_r.sources = r.sources
                results.append(mock_r)

        except Exception as e:
            print(f"[Demo Search] Unified search failed, falling back: {e}")
            global_unified_search = None

    if global_unified_search is None:
        # Fallback to visual-only search
        raw_results = search_index.search_by_text(query, k=20)
        visual_results = [
            r for r in raw_results
            if "[AUDIO]" not in r.camera_id and "Speech:" not in r.camera_id
        ]
        norm_active_path = normalize_video_path(active_video_path)
        visual_results = [r for r in visual_results if normalize_video_path(r.video_path) == norm_active_path]
        
        if anchor_only:
            visual_results = [r for r in visual_results if r.is_anchor]

        audio_results = []
        if global_audio_indexer is not None:
            video_id = os.path.basename(active_video_path)
            audio_results = global_audio_indexer.search(query, video_id=video_id, k=20)

        min_gap = 8.0
        from adve.audio.multimodal_search import merge_results
        merged = merge_results(visual_results, audio_results, min_gap=min_gap, top_k=5)

        if min_similarity > 0.0:
            merged = [r for r in merged if r.similarity >= min_similarity]

        for r in merged[:5]:
            if r.source == "audio" or r.source == "both":
                camera_id = f"{r.video_path} (Speech: \"{r.text}\")"
            else:
                camera_id = getattr(r, "camera_id", r.video_path)
                
            frame_idx = getattr(r, "frame_idx", 0)
            if frame_idx == 0 and r.timestamp > 0:
                frame_idx = int(r.timestamp * 30)

            mock_r = SearchResult(
                video_path = r.video_path,
                camera_id  = camera_id,
                timestamp  = r.timestamp,
                frame_idx  = frame_idx,
                similarity = r.similarity,
                is_anchor  = r.is_anchor,
            )
            mock_r.source = r.source
            mock_r.text_found = ""
            mock_r.audio_text = getattr(r, "text", "")
            mock_r.sources = [r.source] if r.source != "both" else ["visual", "audio"]
            results.append(mock_r)

    active_search_results = results

    # ── Confidence gate ──────────────────────────────────────────────────────
    # CLIP always returns *something* — even for queries completely absent from
    # the video.  We reject results whose best calibrated similarity is below a
    # hard threshold so the UI never shows "100% confident" nonsense answers.
    #
    # Calibration reference (ViT-L/14@336px, cosine sim):
    #   > 0.30  →  strong visual match   (show normally)
    #   0.22–0.30 → weak / uncertain     (show with warning)
    #   < 0.22  →  no meaningful match   (reject)
    STRONG_THRESHOLD = 0.30
    WEAK_THRESHOLD   = 0.22

    _NO_MATCH_OUTPUTS = (
        None,
        [],
        gr.Video(value=None, label="Match 1 Clip", visible=False),
        gr.Video(value=None, label="Match 2 Clip", visible=False),
        gr.Video(value=None, label="Match 3 Clip", visible=False),
    )

    if not results:
        return (
            "### ❌ No Matches Found\n"
            f"The query **\"{query}\"** did not match anything in the indexed video.",
            *_NO_MATCH_OUTPUTS[1:],
        )

    best_sim = results[0].similarity
    if best_sim < WEAK_THRESHOLD:
        return (
            f"### ❌ No Confident Match for \"{query}\"\n"
            f"Best similarity found: **{best_sim*100:.1f}%** — below the confidence threshold.\n\n"
            "This query does not appear in the indexed video. Try a different search term.",
            [],
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False),
        )
        
    # Generate matching visual previews
    previews = []
    os.makedirs("demo_previews", exist_ok=True)
    
    for i, r in enumerate(results):
        cap = cv2.VideoCapture(r.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            preview_path = os.path.join("demo_previews", f"match_{i}_{r.timestamp:.1f}.jpg")
            cv2.imwrite(preview_path, frame)
            previews.append((preview_path, f"{r.timestamp:.1f}s (Match: {r.similarity * 100:.1f}%)"))
            
    # Extract video clips for top 3 results
    clip_paths = [None, None, None]
    labels = ["Match 1 Clip", "Match 2 Clip", "Match 3 Clip"]
    visibilities = [False, False, False]
    
    for i in range(min(3, len(results))):
        r = results[i]
        if use_dynamic_duration:
            dur = get_dynamic_duration(active_video_path, r.timestamp, default_duration=clip_duration)
        else:
            dur = clip_duration
            
        print(f"[Demo Search] Extracting clip {i+1} at {r.timestamp:.1f}s with duration {dur:.1f}s")
        c_path = extract_clip(active_video_path, r.timestamp, duration=dur)
        if c_path and os.path.exists(c_path):
            clip_paths[i] = c_path
            labels[i] = f"Match {i+1}: {r.timestamp:.1f}s (Similarity: {r.similarity * 100:.1f}%, Duration: {dur:.1f}s)"
            visibilities[i] = True
            
    output_text = "### 🔍 Match Results:\n"
    if best_sim < STRONG_THRESHOLD:
        output_text += (
            f"> ⚠️ **Low confidence** — best match is only {best_sim*100:.1f}%. "
            "Results may not be relevant.\n\n"
        )
    for i, r in enumerate(results):
        pct = r.similarity * 100
        output_text += f"{i+1}. **Timestamp: {r.timestamp:.1f}s** | Match: **{pct:.1f}%**"
        
        sources = getattr(r, "sources", None)
        if sources:
            source_labels = []
            if "visual" in sources: source_labels.append("👁 Visual")
            if "ocr" in sources: source_labels.append("📝 Text")
            if "audio" in sources: source_labels.append("🎤 Audio")
            output_text += f" (Sources: {', '.join(source_labels)})"
            
        output_text += f" | Frame {r.frame_idx}\n"
        
        text_found = getattr(r, "text_found", "")
        audio_text = getattr(r, "audio_text", "")
        if text_found:
            output_text += f"   - 📝 **Visible Text**: \"{text_found}\"\n"
        if audio_text:
            output_text += f"   - 🎤 **Spoken Words**: \"{audio_text}\"\n"
        
    return (
        output_text,
        previews,
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )



def chatbot_rag_answer(question: str, history: list):
    """RAG Answer Engine: Send top matching frames and dialog as context to Groq to answer conversational queries."""
    global active_video_path
    if not active_video_path:
        return "Please index a video first."
    if not question.strip():
        return "Please enter a question."
        
    # Load Groq API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ GROQ_API_KEY environment variable is missing.\n\n"
            "Please set your key to enable Groq Video RAG:\n"
            "Windows: `$env:GROQ_API_KEY='your-key'`\n"
            "Linux/macOS: `export GROQ_API_KEY='your-key'`"
        )
        
    # Query database specifically for the question to get question-centric context
    print(f"[Chat RAG] Searching video for question context: '{question}'")
    raw_results = search_index.search_by_text(question, k=15)
    
    # Filter results to the active video, using normalized paths
    norm_active_path = normalize_video_path(active_video_path)
    video_results = [r for r in raw_results if normalize_video_path(r.video_path) == norm_active_path]
    
    # Deduplicate temporally (at least 8.0 seconds apart)
    deduped_results = []
    for r in video_results:
        if not any(abs(r.timestamp - accepted.timestamp) < 8.0 for accepted in deduped_results):
            deduped_results.append(r)
            
    # Slice to top 3 for Groq context
    rag_context_results = deduped_results[:3]
    
    if not rag_context_results:
        return "No relevant moments or transcripts found in the video to answer this question."
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build prior history text to append to system instructions
        history_context = ""
        if history:
            history_context = "Here is the conversation history so far for context:\n"
            for item in history:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    history_context += f"{role.capitalize()}: {content}\n"
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    u_text = item[0]["content"] if isinstance(item[0], dict) else item[0]
                    a_text = item[1]["content"] if isinstance(item[1], dict) else item[1]
                    history_context += f"User: {u_text}\nAssistant: {a_text}\n"
            history_context += "\n"

        prompt_text = (
            f"You are an AI Video RAG Chatbot. Answer the user's current question based on the retrieved video context "
            f"and the conversation history.\n\n"
            f"{history_context}"
            f"Current User Question: '{question}'\n\n"
            f"Use the following retrieved frame images and spoken transcripts to formulate your answer. "
            f"Provide timestamps (e.g. [12.5s]) in your response when referencing specific moments."
        )

        prompt_content = [
            {
                "type": "text", 
                "text": prompt_text
            }
        ]
        
        for i, r in enumerate(rag_context_results):
            w_start = max(0.0, r.timestamp - 5.0)
            w_end = r.timestamp + 10.0
            
            transcript_text = ""
            try:
                cursor = search_index.db.execute(
                    "SELECT timestamp, text FROM transcripts WHERE video_path = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
                    (active_video_path, w_start, w_end)
                )
                rows = cursor.fetchall()
                if rows:
                    transcript_text = " ".join([f"[{ts:.1f}s] {text}" for ts, text in rows])
                else:
                    transcript_text = "(No spoken audio detected in this window)"
            except Exception as e:
                print(f"[Chat RAG] Error querying transcripts: {e}")
                transcript_text = "(Error retrieving transcripts)"

            cap = cv2.VideoCapture(r.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize frame to a max width of 600px to optimize API payload
                h, w = frame.shape[:2]
                if w > 600:
                    scale = 600 / w
                    frame = cv2.resize(frame, (600, int(h * scale)))
                    
                _, buf = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buf).decode()
                
                prompt_content.append({
                    "type": "text",
                    "text": (
                        f"--- Scene Chunk {i+1} at timestamp {r.timestamp:.1f}s (match score: {r.similarity * 100:.1f}%) ---\n"
                        f"Spoken Dialogue Transcript: \"{transcript_text}\""
                    )
                })
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })
                
        prompt_content.append({
            "type": "text",
            "text": "Analyze both the visual details in the frames and the spoken dialogue transcripts to answer the user's question accurately."
        })
        
        print("[Chat RAG] Sending multi-modal query to Groq Llama 4 Scout Vision...")
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_content}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq Llama 3.2 Vision: {e}"


def chatbot_chat_flow(message: str, history: list, clip_duration: float, use_dynamic_duration: bool, min_similarity: float):
    """Handles Chatbot queries, updates conversational history, and refreshes video clip players and preview gallery."""
    if not message.strip():
        return "", history, [], gr.Video(value=None, visible=False), gr.Video(value=None, visible=False), gr.Video(value=None, visible=False)
        
    # 1. Get answer from the conversational RAG engine
    answer = chatbot_rag_answer(message, history)
    
    # 2. Get the visual previews & clips matching the user's message/question to update the UI
    raw_results = search_index.search_by_text(message, k=15)
    norm_active_path = normalize_video_path(active_video_path) if active_video_path else ""
    video_results = [r for r in raw_results if normalize_video_path(r.video_path) == norm_active_path]
    
    # Deduplicate temporally
    deduped_results = []
    for r in video_results:
        if not any(abs(r.timestamp - accepted.timestamp) < 8.0 for accepted in deduped_results):
            deduped_results.append(r)
            
    previews = []
    clip_paths = [None, None, None]
    labels = ["Match 1 Clip", "Match 2 Clip", "Match 3 Clip"]
    visibilities = [False, False, False]
    
    if active_video_path and deduped_results:
        os.makedirs("demo_previews", exist_ok=True)
        for i, r in enumerate(deduped_results[:5]):
            cap = cv2.VideoCapture(r.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                preview_path = os.path.join("demo_previews", f"chat_match_{i}_{r.timestamp:.1f}.jpg")
                cv2.imwrite(preview_path, frame)
                previews.append((preview_path, f"{r.timestamp:.1f}s (Match: {r.similarity * 100:.1f}%)"))
                
        # Extract clips for top 3 matching moments
        for i in range(min(3, len(deduped_results))):
            r = deduped_results[i]
            if use_dynamic_duration:
                dur = get_dynamic_duration(active_video_path, r.timestamp, default_duration=clip_duration)
            else:
                dur = clip_duration
                
            c_path = extract_clip(active_video_path, r.timestamp, duration=dur)
            if c_path and os.path.exists(c_path):
                clip_paths[i] = c_path
                labels[i] = f"Chat Moment {i+1}: {r.timestamp:.1f}s (Similarity: {r.similarity * 100:.1f}%, Duration: {dur:.1f}s)"
                visibilities[i] = True
                
    # 3. Append user message and bot response to chatbot history
    updated_history = list(history) if history else []
    
    # Check history structure (dict vs tuple) or Gradio version
    is_dict_format = True
    if updated_history:
        if isinstance(updated_history[0], (list, tuple)):
            is_dict_format = False
    else:
        try:
            import gradio as gr
            if hasattr(gr, "__version__") and int(gr.__version__.split(".")[0]) < 5:
                is_dict_format = False
        except Exception:
            is_dict_format = False
        
    if is_dict_format:
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": answer})
    else:
        updated_history.append((message, answer))
        
    return (
        "",  # clear the input textbox
        updated_history,
        previews,
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )


def clear_chat():
    return [], "", []


# ── Gradio Theme & Custom CSS ───────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply premium font globally */
* {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background: radial-gradient(circle at top right, #111827, #0b0f17) !important;
    color: #f3f4f6 !important;
}

/* Glassmorphism containers */
.glass-panel {
    background: rgba(17, 24, 39, 0.45) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 20px !important;
}

/* Sidebar and Panel headings */
.pane-title {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 12px !important;
    letter-spacing: -0.01em !important;
}

/* Tabs style custom adjustment */
.tabs {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 6px !important;
}

.tabitem {
    background: transparent !important;
    border: none !important;
    padding: 16px 8px !important;
}

/* Tab button styling */
.tabs button.selected {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}

/* Custom Primary Buttons with Neon Glow */
button.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.2) !important;
}

button.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(1.1) !important;
}

button.secondary-btn {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #e5e7eb !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

button.secondary-btn:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    color: white !important;
}

/* Stats/Widget Container */
.stats-card {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    padding: 18px !important;
}

/* Input Elements formatting */
input, textarea, select {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: #f9fafb !important;
}

input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Badges for confidence metrics */
.badge {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    padding: 2px 8px !important;
    border-radius: 9999px !important;
    font-size: 0.75rem !important;
}
"""

with gr.Blocks(title="ADVE Engine Portal", css=custom_css) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 24px; padding-top: 15px;">
        <h1 style="font-size: 36px; font-weight: 800; background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px;">ADVE — Semantic Video RAG & Chatbot</h1>
        <p style="font-size: 15px; color: #9ca3af; max-width: 850px; margin: 0 auto; line-height: 1.5;">
            Anchor-Delta Video Embedding (ADVE) reduces neural vision network calls by up to 90% via motion-adaptive keyframe processing, facilitating fast indexing, zero-latency semantic search, and conversational RAG.
        </p>
    </div>
    """)
    
    with gr.Row():
        # --- LEFT PANE: Inputs, Search & Chatbot ---
        with gr.Column(scale=5, elem_classes="glass-panel"):
            gr.HTML("<h2 class='pane-title'>📥 1. Video Ingestion & Indexing</h2>")
            
            with gr.Tabs(elem_classes="tabs"):
                # --- YouTube Tab ---
                with gr.TabItem("YouTube Video Link"):
                    yt_url = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=aircAruvnKk")
                    with gr.Row():
                        yt_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1)
                        yt_adaptive = gr.Checkbox(label="Adaptive FPS (Motion-Based)", value=True)
                    with gr.Row():
                        yt_index_audio = gr.Checkbox(label="Index Dialogue (Whisper Audio)", value=False, info="Required for dialogue RAG search. (Slow on CPUs)")
                        yt_index_ocr = gr.Checkbox(label="Index Screen Text (EasyOCR)", value=False, info="Required for on-screen text search. (Slow on CPUs)")
                    yt_index_btn = gr.Button("Download & Index Video", variant="primary", elem_classes="primary-btn")
                    yt_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Paste a link and click Index...")
                    
                # --- Local File Tab ---
                with gr.TabItem("Local Video File"):
                    local_file = gr.File(label="Upload MP4 / WebM File", file_types=["video"])
                    with gr.Row():
                        local_fps = gr.Slider(label="Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1)
                        local_adaptive = gr.Checkbox(label="Adaptive FPS (Motion-Based)", value=True)
                    with gr.Row():
                        local_index_audio = gr.Checkbox(label="Index Dialogue (Whisper Audio)", value=False, info="Required for dialogue RAG search. (Slow on CPUs)")
                        local_index_ocr = gr.Checkbox(label="Index Screen Text (EasyOCR)", value=False, info="Required for on-screen text search. (Slow on CPUs)")
                    local_index_btn = gr.Button("Index Uploaded Video", variant="primary", elem_classes="primary-btn")
                    local_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Upload a file and click Index...")

            gr.HTML("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0;'></div>")

            gr.HTML("<h2 class='pane-title'>🔍 2. Semantic Scene Search</h2>")
            search_query = gr.Textbox(label="Search Query", placeholder="e.g. 'pedestrians crossing road', 'concept of neural networks'")
            with gr.Row():
                clip_duration = gr.Slider(label="Clip Duration (seconds)", minimum=3.0, maximum=60.0, value=10.0, step=1.0)
                min_similarity = gr.Slider(label="Min Similarity Gate", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            with gr.Row():
                use_dynamic_duration = gr.Checkbox(label="Auto-Detect Scene Duration", value=False)
                anchor_only = gr.Checkbox(label="Search Anchor Frames Only", value=False)
            search_btn = gr.Button("Search Video", variant="primary", elem_classes="primary-btn")
            search_metrics = gr.Markdown("No query submitted yet.")

            gr.HTML("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0;'></div>")

            gr.HTML("<h2 class='pane-title'>🤖 3. Conversational Video Chatbot</h2>")
            chatbot = gr.Chatbot(label="ADVE Video Chatbot", height=320)
            chat_input = gr.Textbox(label="Ask a question about the video...", placeholder="e.g. 'What is the presenter writing at 2:30?'")
            with gr.Row():
                chat_submit = gr.Button("Send Question", variant="primary", elem_classes="primary-btn")
                chat_clear_btn = gr.Button("Clear Chat", elem_classes="secondary-btn")

        # --- RIGHT PANE: Consolidated Synced Outputs & Playback ---
        with gr.Column(scale=5, elem_classes="glass-panel"):
            gr.HTML("<h2 class='pane-title'>🎞️ Synced Media Console</h2>")
            
            # Matched Frame Anchors Gallery
            gr.HTML("<h3 style='font-size: 14px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>Matching Keyframe Anchors</h3>")
            gallery_previews = gr.Gallery(label="Matching Frame Anchors", columns=3, rows=2, object_fit="contain", height=240)
            
            gr.HTML("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0;'></div>")
            
            # Video Playback Clip Players
            gr.HTML("<h3 style='font-size: 14px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>Extracted Match Playback Clips</h3>")
            with gr.Tabs(elem_classes="tabs"):
                with gr.TabItem("Match 1"):
                    clip_player_1 = gr.Video(label="Match 1 Clip", visible=False)
                with gr.TabItem("Match 2"):
                    clip_player_2 = gr.Video(label="Match 2 Clip", visible=False)
                with gr.TabItem("Match 3"):
                    clip_player_3 = gr.Video(label="Match 3 Clip", visible=False)
                    
            gr.HTML("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0;'></div>")
            
            # Performance Metric / Quick Guide Card
            with gr.Column(elem_classes="stats-card"):
                gr.HTML("<h4 style='font-weight: 600; color: #cbd5e1; margin-top: 0;'>💡 System Instructions & Guidance</h4>")
                gr.Markdown(
                    "1. **Super-fast Ingestion:** Default options bypass slow OCR and Whisper layers. Visual-only indexing runs in seconds (up to **8x faster**).\n"
                    "2. **Search / Chat Integration:** Submitting a search or chatting with the bot automatically updates the right-hand column's keyframe gallery and playbacks simultaneously.\n"
                    "3. **Audio Indexing Option:** If you need dialog search, check 'Index Dialogue (Whisper Audio)' during ingestion."
                )

    # ── Button Bindings ──────────────────────────────────────────────────────────
    yt_index_btn.click(
        fn=handle_youtube_index,
        inputs=[yt_url, yt_fps, yt_adaptive, yt_index_audio, yt_index_ocr],
        outputs=[yt_status]
    )
    
    local_index_btn.click(
        fn=handle_local_index,
        inputs=[local_file, local_fps, local_adaptive, local_index_audio, local_index_ocr],
        outputs=[local_status]
    )
    
    search_btn.click(
        fn=search_and_retrieve,
        inputs=[search_query, clip_duration, use_dynamic_duration, anchor_only, min_similarity],
        outputs=[search_metrics, gallery_previews, clip_player_1, clip_player_2, clip_player_3]
    )
    
    # Chatbot submit bindings (Submit on Send or enter key)
    chat_submit.click(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, gallery_previews, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_input.submit(
        fn=chatbot_chat_flow,
        inputs=[chat_input, chatbot, clip_duration, use_dynamic_duration, min_similarity],
        outputs=[chat_input, chatbot, gallery_previews, clip_player_1, clip_player_2, clip_player_3]
    )
    chat_clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, chat_input, gallery_previews]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch demo with a public shareable URL")
    args = parser.parse_args()
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share,
                theme=gr.themes.Monochrome(), css=custom_css)
