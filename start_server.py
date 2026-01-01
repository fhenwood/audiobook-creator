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
    uvicorn.run("web_app:app", host="0.0.0.0", port=7860, access_log=True)

def run_orpheus_tts():
    """Start the Orpheus TTS API server on port 8880"""
    # Add containers/orpheus to path
    sys.path.insert(0, "/app/containers/orpheus")
    os.chdir("/app/containers/orpheus")
    
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8880, access_log=True)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutting down servers...")
    sys.exit(0)

def run_tests():
    """Run E2E integration tests."""
    print("\n🧪 Running Integrity Tests...")
    try:
        # Run specific robustness tests
        result = subprocess.run(
            ["pytest", "tests/integration/test_pipeline_robustness.py", "-v"],
            check=True
        )
        print("✅ Integrity Tests Passed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Integrity Tests FAILED!")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run integration tests on startup")
    args = parser.parse_args()

    print("🚀 Starting Audiobook Creator Services...")
    print("=" * 50)
    
    if args.test or os.environ.get("RUN_TESTS") == "1":
        if not run_tests():
            print("⚠️ Startup aborted due to test failure.")
            sys.exit(1)
        print("=" * 50)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run Gradio in main process
    print("✅ Gradio Web UI starting on port 7860")
    print("=" * 50)
    print("🌐 Web UI: http://localhost:7860")
    print("🔊 TTS API: http://localhost:7860/api")
    print("=" * 50)
    
    try:
        run_gradio()
    except KeyboardInterrupt:
        pass
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
