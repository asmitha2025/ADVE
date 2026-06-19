import os
import sys
import subprocess
import time
import webbrowser

def print_banner():
    print("=" * 65)
    print("        ADVE v2.0 - Anchor-Delta Video Embedding Engine")
    print("=" * 65)
    print("Welcome to the ADVE Product Suite Demo!")
    print("\nThis demo runs the entire system locally:")
    print("  1. Public Product Landing Page (ROI Calculator, User Registration)")
    print("  2. App Dashboard Workspace (Video Upload, Vector Indexing, Search)")
    print("  3. Whisper Speech Transcription Integration (Optional)")
    print("  4. Learned MLP Embedding Reconstruction Pipeline")
    print("=" * 65)

def check_whisper():
    try:
        import whisper
        print("✓ Whisper is installed.")
        return True
    except ImportError:
        print("! Whisper is not installed.")
        print("To enable Whisper audio transcription, run: pip install openai-whisper")
        print("Proceeding without Whisper (speech search will be mocked/disabled).")
        return False

def start_server():
    print("\nStarting FastAPI server on http://localhost:8000/...")
    
    # Configure path to include adve_v2
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("adve_v2") + os.pathsep + env.get("PYTHONPATH", "")
    
    # Run uvicorn server in a subprocess
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "adve.api.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    process = subprocess.Popen(cmd, env=env)
    
    # Wait for server to start
    time.sleep(3)
    return process

def main():
    print_banner()
    check_whisper()
    
    server_process = None
    try:
        server_process = start_server()
        
        print("\n" + "-" * 50)
        print("ADVE Platform is live! Running on port 8000.")
        print("Opening http://localhost:8000/ in your browser...")
        print("Press Ctrl+C in this terminal to stop the server.")
        print("-" * 50)
        
        webbrowser.open("http://localhost:8000/")
        
        # Keep the script running to keep the server alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping ADVE server...")
    finally:
        if server_process:
            server_process.terminate()
            print("Server stopped.")

if __name__ == "__main__":
    main()
