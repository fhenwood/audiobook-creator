"""
Background Task Runner - Runs audiobook generation jobs independently of Gradio connection.

This allows users to start a job, close their browser, and come back later to check on progress
or download the completed audiobook.
"""

import asyncio
import threading
import traceback
from typing import Dict, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class BackgroundTaskRunner:
    """
    Singleton that manages background audiobook generation tasks.
    Jobs run in a background thread/event loop, independent of the Gradio WebSocket connection.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # Only 1 job at a time (GPU constraint)
        self._start_background_loop()
    
    def _start_background_loop(self):
        """Start a background event loop for running async tasks."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            print("🔄 Background task runner started")
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True, name="background-tasks")
        self._thread.start()
        
        # Wait for loop to be ready
        import time
        for _ in range(50):
            if self._loop is not None:
                break
            time.sleep(0.1)
    
    def submit_job(self, job_id: str, coro_factory: Callable) -> bool:
        """
        Submit a job to run in the background.
        
        Args:
            job_id: The job ID for tracking
            coro_factory: A callable that returns a coroutine when called.
                         This is called in the background loop, not the main thread.
        
        Returns:
            True if job was submitted, False if a job is already running
        """
        if self._loop is None:
            print("❌ Background loop not ready")
            return False
        
        # Check if a job is already running (GPU constraint - only one at a time)
        for existing_job_id, task in list(self._running_jobs.items()):
            if not task.done():
                print(f"⚠️ Cannot start job {job_id}: Job {existing_job_id} is already running")
                return False
        
        async def run_job():
            try:
                print(f"🚀 Starting background job {job_id}")
                # Create the coroutine in the background loop
                coro = coro_factory()
                # Await the coroutine to completion (it's a regular async function, not a generator)
                await coro
                print(f"✅ Background job {job_id} completed")
            except Exception as e:
                print(f"❌ Background job {job_id} failed: {e}")
                traceback.print_exc()
            finally:
                if job_id in self._running_jobs:
                    del self._running_jobs[job_id]
        
        # Schedule the job in the background loop
        future = asyncio.run_coroutine_threadsafe(run_job(), self._loop)
        # Store a handle (we'll wrap it in a fake task for tracking)
        self._running_jobs[job_id] = _FutureTaskWrapper(future)
        
        return True
    
    def is_job_running(self, job_id: str) -> bool:
        """Check if a specific job is currently running."""
        if job_id in self._running_jobs:
            task = self._running_jobs[job_id]
            return not task.done()
        return False
    
    def get_running_jobs(self) -> list:
        """Get list of currently running job IDs."""
        return [
            job_id for job_id, task in self._running_jobs.items()
            if not task.done()
        ]
    
    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a running job."""
        if job_id in self._running_jobs:
            task = self._running_jobs[job_id]
            if not task.done():
                task.cancel()
                print(f"⏹️ Cancelled job {job_id}")
                return True
        return False


class _FutureTaskWrapper:
    """Wrapper to make a Future look like a Task for our tracking."""
    def __init__(self, future):
        self._future = future
    
    def done(self):
        return self._future.done()
    
    def cancel(self):
        return self._future.cancel()


# Global instance
background_runner = BackgroundTaskRunner()
