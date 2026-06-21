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

# Ensure adve_v2 is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config
from adve.search.index import ADVESearchIndex, SearchResult

# Global state to keep track of active index, video, and search results
active_video_path = None
active_search_results = []
index_dir = os.path.join(current_dir, "data", "demo_index")
os.makedirs(index_dir, exist_ok=True)
search_index = ADVESearchIndex(index_dir)


def get_ffmpeg_binary() -> Optional[str]:
    """Find a working ffmpeg binary on the system (PATH or imageio-ffmpeg)."""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
        
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def is_ffmpeg_available() -> bool:
    return get_ffmpeg_binary() is not None



def download_youtube(url: str, progress=gr.Progress()) -> str:
    """Download the first 5 minutes of a YouTube video, or fall back to low-res full download if ffmpeg is missing."""
    progress(0.1, desc="Checking YouTube URL and system capabilities...")
    os.makedirs("demo_videos", exist_ok=True)
    
    ffmpeg_ok = is_ffmpeg_available()
    
    if ffmpeg_ok:
        # We limit duration to first 5 minutes (300 seconds) to ensure quick demo download & processing
        opts = {
            "format": "mp4/best",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": 300}],
            "force_keyframes_at_cuts": True,
            "quiet": True,
            "no_warnings": True,
        }
        progress(0.3, desc="Downloading video from YouTube (partial download first 5m)...")
    else:
        # Without ffmpeg, download full video at lowest quality/resolution (very small size) to save time/bandwidth
        opts = {
            "format": "worst[ext=mp4]/mp4",
            "outtmpl": "demo_videos/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
        }
        progress(0.3, desc="ffmpeg not found. Downloading full video in low-res format to save bandwidth...")
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            ext = info.get("ext", "mp4")
            expected_path = f"demo_videos/{video_id}.{ext}"
            
            if os.path.exists(expected_path):
                return expected_path
            
            # Scan directory as fallback
            for file in os.listdir("demo_videos"):
                if file.startswith(video_id):
                    return os.path.join("demo_videos", file)
            raise FileNotFoundError("Downloaded video file not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {e}")


def index_video(video_path: str, sampling_rate: float = 5.0, progress=gr.Progress()) -> str:
    """Run ADVE pipeline to index anchor frames in the video."""
    global active_video_path
    active_video_path = video_path

    progress(0.0, desc="Initializing ADVE pipeline...")
    config = Config()
    # CLIP runs on CPU (for memory stability) while YOLO runs on GPU (for speed) if CUDA is available
    pipeline = ADVEPipeline(config)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Calculate step size based on sampling rate (e.g. 1 FPS means processing 1 frame every 'fps' frames)
    frame_step = max(1, int(fps / sampling_rate))
    print(f"[Demo Indexer] Total frames to process: {total_frames} @ {fps:.1f} FPS")
    print(f"[Demo Indexer] Sampling rate: {sampling_rate} FPS (processing 1 frame every {frame_step} frames)")

    # Clear out any previous database entries to keep the demo clean
    global search_index
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

    
    cap = cv2.VideoCapture(video_path)
    idx = 0
    sampled_idx = 0
    anchors_count = 0
    start_time = time.time()
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx % frame_step == 0:
            result = pipeline.process_frame(frame, idx, no_validation=True)
            
            # Save only ANCHOR frames for RAG chunking
            if result["is_anchor"]:
                timestamp = idx / fps
                search_index.add(video_path, "youtube_cam", timestamp, idx, result["embedding"], is_anchor=True)
                anchors_count += 1
            sampled_idx += 1
            
        idx += 1
        if idx % 15 == 0:
            progress_pct = min(0.99, idx / max(1, total_frames))
            progress(progress_pct, desc=f"Ingested {idx}/{total_frames} frames (Processed {sampled_idx} sampled frames)")

    cap.release()
    search_index.save()
    elapsed = time.time() - start_time
    
    savings = 100.0 * (1.0 - (anchors_count / max(1, sampled_idx)))
    summary = (
        f"✅ Indexing Complete!\n"
        f"- Total Video Frames: {idx}\n"
        f"- Sampled Frames Processed: {sampled_idx} (sampling at {sampling_rate} FPS)\n"
        f"- RAG Anchor Chunks Created: {anchors_count}\n"
        f"- ADVE Neural Cost Savings: {savings:.1f}%\n"
        f"- Processing Time: {elapsed:.2f} seconds ({sampled_idx/elapsed:.1f} FPS)"
    )
    return summary


def handle_youtube_index(url: str, sampling_rate: float, progress=gr.Progress()) -> str:
    if not url.strip():
        return "Please enter a valid YouTube URL."
    try:
        video_path = download_youtube(url, progress)
        return index_video(video_path, sampling_rate, progress)
    except Exception as e:
        return f"Error: {e}"


def handle_local_index(file, sampling_rate: float, progress=gr.Progress()) -> str:
    if file is None:
        return "Please upload a video file first."
    return index_video(file, sampling_rate, progress)


def extract_clip(video_path: str, timestamp: float, duration: float = 10.0) -> str:
    """Extract a video clip around the timestamp. Falls back to OpenCV if ffmpeg is missing."""
    os.makedirs("clips", exist_ok=True)
    start = max(0.0, timestamp - 2.0)
    output_path = os.path.join("clips", f"clip_{timestamp:.1f}.mp4")
    
    # Check if ffmpeg is available
    ffmpeg_bin = get_ffmpeg_binary()
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


def search_and_retrieve(query: str, clip_duration: float, use_dynamic_duration: bool):
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
    results = search_index.search_by_text(query, k=5)
    active_search_results = results
    
    if not results:
        return (
            "No matches found.", 
            [], 
            gr.Video(value=None, label="Match 1 Clip", visible=False),
            gr.Video(value=None, label="Match 2 Clip", visible=False),
            gr.Video(value=None, label="Match 3 Clip", visible=False)
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
            
    output_text = "### Match Results:\n"
    for i, r in enumerate(results):
        output_text += f"{i+1}. **Timestamp: {r.timestamp:.1f}s** (Frame {r.frame_idx}) | Similarity: **{r.similarity * 100:.1f}%**\n"
        
    return (
        output_text,
        previews,
        gr.Video(value=clip_paths[0], label=labels[0], visible=visibilities[0]),
        gr.Video(value=clip_paths[1], label=labels[1], visible=visibilities[1]),
        gr.Video(value=clip_paths[2], label=labels[2], visible=visibilities[2])
    )



def rag_answer_question(question: str):
    """RAG Answer Engine: Send top matching frames as context to Groq Llama 3.2 Vision to answer queries."""
    global active_search_results, active_video_path
    if not active_video_path:
        return "Please index a video first."
    if not active_search_results:
        return "Please perform a search first to retrieve video frames."
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
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build prompt content with base64 visual frames
        prompt_content = [
            {
                "type": "text", 
                "text": f"You are an AI Video RAG agent explaining scenes from a video. Answer the user's question: '{question}' using the relevant retrieved frames below."
            }
        ]
        
        for i, r in enumerate(active_search_results[:3]):
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
                    "text": f"Retrieved Frame {i+1} at timestamp {r.timestamp:.1f}s (match score: {r.similarity * 100:.1f}%):"
                })
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })
                
        prompt_content.append({
            "type": "text",
            "text": "Analyze the visual details in the frames and answer the question concisely."
        })
        
        print("[Demo RAG] Sending multi-modal query to Groq Llama 3.2 Vision...")
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": prompt_content}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying Groq Llama 3.2 Vision: {e}"


# ── Gradio Theme & Custom CSS ───────────────────────────────────────────────────

custom_css = """
body { background-color: #0b0d11 !important; color: #f3f4f6 !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.tabs { background: rgba(20, 26, 36, 0.5) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
.tabitem { padding: 20px !important; }
button.primary { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; border: none !important; color: white !important; font-weight: 600 !important; }
button.primary:hover { filter: brightness(1.1) !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px; padding-top: 20px;">
        <h1 style="font-size: 32px; font-weight: 800; background: linear-gradient(to right, #3b82f6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;">ADVE — Semantic Video Search & RAG</h1>
        <p style="font-size: 16px; color: #9ca3af; max-width: 800px; margin: 0 auto;">Anchor-Delta Video Embedding processes only scene changes, reducing neural vision network calls by up to 90% while building high-accuracy semantic search indices.</p>
    </div>
    """)
    
    with gr.Tabs():
        # --- YouTube Tab ---
        with gr.TabItem("YouTube Video Search"):
            with gr.Row():
                with gr.Column(scale=2):
                    yt_url = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=aircAruvnKk")
                    yt_fps = gr.Slider(label="Frame Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1, info="Frames to process per second. 5.0 FPS is recommended for standard RAG indexing.")
                    yt_index_btn = gr.Button("Download & Index Video", variant="primary")
                with gr.Column(scale=3):
                    yt_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Paste a link and click Index...")
                    
        # --- Local File Tab ---
        with gr.TabItem("Local Video Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    local_file = gr.File(label="Upload MP4 / WebM File", file_types=["video"])
                    local_fps = gr.Slider(label="Frame Sampling Rate (FPS)", minimum=0.1, maximum=10.0, value=5.0, step=0.1, info="Frames to process per second. 5.0 FPS is recommended for standard RAG indexing.")
                    local_index_btn = gr.Button("Index Uploaded Video", variant="primary")
                with gr.Column(scale=3):
                    local_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Upload a file and click Index...")
                    
    # --- Unified Search & RAG Section ---
    with gr.Row(visible=True) as search_row:
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Semantic Scene Search")
            search_query = gr.Textbox(label="Query (e.g. 'pedestrians crossing road', 'concept of neural networks')", placeholder="What are you looking for?")
            clip_duration = gr.Slider(label="Clip Duration (seconds)", minimum=3.0, maximum=60.0, value=10.0, step=1.0, info="Length of the extracted clip")
            use_dynamic_duration = gr.Checkbox(label="Auto-Detect Scene Duration (Dynamic)", value=False, info="Automatically compute clip length using scene/anchor boundaries")
            search_btn = gr.Button("Search Video", variant="primary")
            search_metrics = gr.Markdown("No query submitted yet.")
            
        with gr.Column(scale=3):
            gr.Markdown("### 🎞️ Match Preview & Clip Playback")
            with gr.Column():
                clip_player_1 = gr.Video(label="Match 1 Clip", visible=False)
                clip_player_2 = gr.Video(label="Match 2 Clip", visible=False)
                clip_player_3 = gr.Video(label="Match 3 Clip", visible=False)
            gallery_previews = gr.Gallery(label="Matching Frame Anchors", columns=3, rows=2, object_fit="contain")

    with gr.Row(visible=True) as rag_row:
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Video RAG Question Engine")
            rag_query = gr.Textbox(label="Ask a question about the video", placeholder="e.g. 'What is the speaker drawing on the board?'")
            rag_btn = gr.Button("Ask Groq (Llama 3.2)", variant="primary")
            
        with gr.Column(scale=3):
            gr.Markdown("### 📝 Groq's Visual Analysis Answer")
            rag_answer = gr.Textbox(label="Groq Visual RAG Answer", interactive=False, lines=8)

    # ── Button Bindings ──────────────────────────────────────────────────────────
    yt_index_btn.click(
        fn=handle_youtube_index,
        inputs=[yt_url, yt_fps],
        outputs=[yt_status]
    )
    
    local_index_btn.click(
        fn=handle_local_index,
        inputs=[local_file, local_fps],
        outputs=[local_status]
    )
    
    search_btn.click(
        fn=search_and_retrieve,
        inputs=[search_query, clip_duration, use_dynamic_duration],
        outputs=[search_metrics, gallery_previews, clip_player_1, clip_player_2, clip_player_3]
    )
    
    rag_btn.click(
        fn=rag_answer_question,
        inputs=[rag_query],
        outputs=[rag_answer]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Launch demo with a public shareable URL")
    args = parser.parse_args()
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
