import os
import subprocess
import tempfile
from typing import List, Dict

class AudioTranscriber:
    """
    Extracts and transcribes audio from video files using Whisper.
    Gracefully handles cases where ffmpeg or whisper are not installed.
    """

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self.whisper_available = False
        self._model = None

        # Check if whisper is installed
        try:
            import whisper
            self.whisper_available = True
        except ImportError:
            print("[AudioTranscriber] openai-whisper package not installed. Speech indexing will be skipped.")

    def _load_model(self):
        if self._model is None and self.whisper_available:
            try:
                import whisper
                print(f"[AudioTranscriber] Loading Whisper model '{self.model_name}'...")
                self._model = whisper.load_model(self.model_name, device="cpu")
                print("[AudioTranscriber] Whisper model loaded successfully.")
            except Exception as e:
                print(f"[AudioTranscriber] Failed to load Whisper model: {e}")
                self.whisper_available = False

    def is_ffmpeg_available(self) -> bool:
        try:
            # Check if ffmpeg is in path by running it with -version
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Extracts audio and transcribes it.
        Returns a list of dicts: [{'timestamp': float, 'text': str}]
        """
        if not self.whisper_available:
            return []

        if not self.is_ffmpeg_available():
            print("[AudioTranscriber] ffmpeg is not available on this system. Audio extraction skipped.")
            return []

        self._load_model()
        if self._model is None:
            return []

        temp_wav = tempfile.mktemp(suffix=".wav")
        try:
            # Extract mono audio at 16kHz using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav
            ]
            print(f"[AudioTranscriber] Extracting audio to {temp_wav}...")
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result.returncode != 0 or not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 100:
                # Video probably has no audio stream or extraction failed
                print("[AudioTranscriber] Audio extraction failed (video may be silent).")
                return []

            print("[AudioTranscriber] Transcribing audio with Whisper...")
            transcription = self._model.transcribe(temp_wav)
            
            segments = []
            for seg in transcription.get("segments", []):
                segments.append({
                    "timestamp": float(seg.get("start", 0.0)),
                    "text": seg.get("text", "").strip()
                })
            
            print(f"[AudioTranscriber] Transcribed {len(segments)} segments.")
            return segments

        except Exception as e:
            print(f"[AudioTranscriber] Error during transcription: {e}")
            return []
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
