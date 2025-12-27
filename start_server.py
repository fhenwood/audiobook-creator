#!/usr/bin/env python3
"""
Audiobook Creator - Combined Server Startup
Runs both Gradio web UI (port 7860) and Orpheus TTS API (port 8880)
"""

import os
import sys
import subprocess
import signal
import time
from multiprocessing import Process

def run_gradio():
    """Start the Gradio web interface on port 7860"""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, access_log=True)

def run_orpheus_tts():
    """Start the Orpheus TTS API server on port 8880"""
    # Add orpheus_tts to path
    sys.path.insert(0, "/app/orpheus_tts")
    os.chdir("/app/orpheus_tts")
    
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8880, access_log=True)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    print("🚀 Starting Audiobook Creator Services...")
    print("=" * 50)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Orpheus TTS in a subprocess
    orpheus_process = Process(target=run_orpheus_tts, name="orpheus-tts")
    orpheus_process.start()
    print(f"✅ Orpheus TTS API starting on port 8880 (PID: {orpheus_process.pid})")
    
    # Give Orpheus time to start
    time.sleep(2)
    
    # Run Gradio in main process
    print("✅ Gradio Web UI starting on port 7860")
    print("=" * 50)
    print("🌐 Web UI: http://localhost:7860")
    print("🔊 TTS API: http://localhost:8880")
    print("=" * 50)
    
    try:
        run_gradio()
    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Stopping Orpheus TTS...")
        orpheus_process.terminate()
        orpheus_process.join(timeout=5)
        if orpheus_process.is_alive():
            orpheus_process.kill()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
