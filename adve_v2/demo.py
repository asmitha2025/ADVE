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


def download_youtube(url: str, progress=gr.Progress()) -> str:
    """Download the first 5 minutes of a YouTube video using yt-dlp."""
    progress(0.1, desc="Checking YouTube URL...")
    os.makedirs("demo_videos", exist_ok=True)
    
    # We limit duration to first 5 minutes (300 seconds) to ensure quick demo download & processing
    opts = {
        "format": "mp4/best",
        "outtmpl": "demo_videos/%(id)s.%(ext)s",
        "download_ranges": lambda info, ydl: [{"start_time": 0, "end_time": 300}],
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        progress(0.3, desc="Downloading video from YouTube (limited to first 5m)...")
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


def index_video(video_path: str, progress=gr.Progress()) -> str:
    """Run ADVE pipeline to index anchor frames in the video."""
    global active_video_path
    active_video_path = video_path

    progress(0.0, desc="Initializing ADVE pipeline...")
    config = Config()
    # CPU is highly robust for the demo, avoiding CUDA DLL paging crashes
    config.DEVICE = "cpu"
    pipeline = ADVEPipeline(config)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    print(f"[Demo Indexer] Total frames to process: {total_frames} @ {fps:.1f} FPS")

    # Clear out any previous database entries to keep the demo clean
    try:
        if os.path.exists(os.path.join(index_dir, "embeddings.faiss")):
            os.remove(os.path.join(index_dir, "embeddings.faiss"))
        if os.path.exists(os.path.join(index_dir, "metadata.db")):
            os.remove(os.path.join(index_dir, "metadata.db"))
    except Exception as e:
        print(f"Warning: Failed to clear old database: {e}")
        
    global search_index
    search_index = ADVESearchIndex(index_dir)
    
    cap = cv2.VideoCapture(video_path)
    idx = 0
    anchors_count = 0
    start_time = time.time()
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        result = pipeline.process_frame(frame, idx, no_validation=True)
        
        # Save only ANCHOR frames for RAG chunking
        if result["is_anchor"]:
            timestamp = idx / fps
            search_index.add(video_path, "youtube_cam", timestamp, idx, result["embedding"], is_anchor=True)
            anchors_count += 1
            
        idx += 1
        if idx % 15 == 0:
            progress_pct = min(0.99, idx / max(1, total_frames))
            progress(progress_pct, desc=f"Ingested {idx}/{total_frames} frames (Anchors created: {anchors_count})")

    cap.release()
    search_index.save()
    elapsed = time.time() - start_time
    
    savings = 100.0 * (1.0 - (anchors_count / max(1, idx)))
    summary = (
        f"✅ Indexing Complete!\n"
        f"- Total Video Frames: {idx}\n"
        f"- RAG Anchor Chunks Created: {anchors_count}\n"
        f"- ADVE Neural Cost Savings: {savings:.1f}%\n"
        f"- Processing Time: {elapsed:.2f} seconds ({idx/elapsed:.1f} FPS)"
    )
    return summary


def handle_youtube_index(url: str, progress=gr.Progress()) -> str:
    if not url.strip():
        return "Please enter a valid YouTube URL."
    try:
        video_path = download_youtube(url, progress)
        return index_video(video_path, progress)
    except Exception as e:
        return f"Error: {e}"


def handle_local_index(file, progress=gr.Progress()) -> str:
    if file is None:
        return "Please upload a video file first."
    return index_video(file, progress)


def extract_clip(video_path: str, timestamp: float, duration: float = 10.0) -> str:
    """Extract a 10s video clip around the timestamp."""
    os.makedirs("clips", exist_ok=True)
    start = max(0.0, timestamp - 2.0)
    output_path = os.path.join("clips", f"clip_{timestamp:.1f}.mp4")
    
    # Try browser-friendly h264 re-encoding first
    cmd = [
        "ffmpeg", "-y",
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
            "ffmpeg", "-y",
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
            return ""


def search_and_retrieve(query: str):
    """Search natural language and return matching frame images and video clips."""
    global active_search_results, active_video_path
    if not active_video_path:
        return "No video has been indexed yet. Please index a video first.", [], None
        
    if not query.strip():
        return "Please enter a search query.", [], None
        
    print(f"[Demo Search] Querying: '{query}'")
    results = search_index.search_by_text(query, k=5)
    active_search_results = results
    
    if not results:
        return "No matches found.", [], None
        
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
            
    # Extract video clip for the top result
    top_result = results[0]
    clip_path = extract_clip(active_video_path, top_result.timestamp)
    
    output_text = "### Match Results:\n"
    for i, r in enumerate(results):
        output_text += f"{i+1}. **Timestamp: {r.timestamp:.1f}s** (Frame {r.frame_idx}) | Similarity: **{r.similarity * 100:.1f}%**\n"
        
    return output_text, previews, clip_path


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
                    yt_index_btn = gr.Button("Download & Index Video", variant="primary")
                with gr.Column(scale=3):
                    yt_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Paste a link and click Index...")
                    
        # --- Local File Tab ---
        with gr.TabItem("Local Video Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    local_file = gr.File(label="Upload MP4 / WebM File", file_types=["video"])
                    local_index_btn = gr.Button("Index Uploaded Video", variant="primary")
                with gr.Column(scale=3):
                    local_status = gr.Textbox(label="Status / Progress Summary", interactive=False, placeholder="Upload a file and click Index...")
                    
    # --- Unified Search & RAG Section ---
    with gr.Row(visible=True) as search_row:
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Semantic Scene Search")
            search_query = gr.Textbox(label="Query (e.g. 'pedestrians crossing road', 'concept of neural networks')", placeholder="What are you looking for?")
            search_btn = gr.Button("Search Video", variant="primary")
            search_metrics = gr.Markdown("No query submitted yet.")
            
        with gr.Column(scale=3):
            gr.Markdown("### 🎞️ Match Preview & Clip Playback")
            top_clip_player = gr.Video(label="Top Match 10-Second Clip")
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
        inputs=[yt_url],
        outputs=[yt_status]
    )
    
    local_index_btn.click(
        fn=handle_local_index,
        inputs=[local_file],
        outputs=[local_status]
    )
    
    search_btn.click(
        fn=search_and_retrieve,
        inputs=[search_query],
        outputs=[search_metrics, gallery_previews, top_clip_player]
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
