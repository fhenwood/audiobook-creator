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

from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


class BackgroundTaskRunner:
    """
    Singleton that manages background audiobook generation tasks.
    Jobs run in a background thread/event loop, independent of the Gradio WebSocket connection.
    
    Concurrency is controlled by settings.max_concurrent_jobs (default: 1 due to GPU constraints).
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
        
        # Load max concurrent jobs from config
        try:
            from audiobook.config import settings
            self._max_concurrent_jobs = settings.max_concurrent_jobs
        except Exception:
            self._max_concurrent_jobs = 1  # Default fallback
        
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent_jobs)
        self._start_background_loop()
        logger.info(f"Background runner initialized with max_concurrent_jobs={self._max_concurrent_jobs}")
    
    @property
    def max_concurrent_jobs(self) -> int:
        """Get the maximum number of concurrent jobs allowed."""
        return self._max_concurrent_jobs
    
    @property
    def active_job_count(self) -> int:
        """Get the number of currently running jobs."""
        return len([j for j in self._running_jobs.values() if not j.done()])
    
    @property
    def can_accept_job(self) -> bool:
        """Check if we can accept another job based on concurrency limit."""
        return self.active_job_count < self._max_concurrent_jobs
    
    def _start_background_loop(self):
        """Start a background event loop for running async tasks."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info("Background task runner started")
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
            True if job was submitted, False if at concurrency limit
        """
        if self._loop is None:
            logger.error("Background loop not ready")
            return False
        
        # Check if we're at the concurrency limit
        if not self.can_accept_job:
            running = self.get_running_jobs()
            logger.warning(f"Cannot start job {job_id}: At concurrency limit ({self._max_concurrent_jobs}). Running: {running}")
            return False
        
        async def run_job():
            try:
                logger.info(f"Starting background job {job_id} ({self.active_job_count + 1}/{self._max_concurrent_jobs} slots)")
                # Create the coroutine in the background loop
                coro = coro_factory()
                # Await the coroutine to completion (it's a regular async function, not a generator)
                await coro
                logger.info(f"Background job {job_id} completed")
            except Exception as e:
                logger.error(f"Background job {job_id} failed: {e}")
                logger.debug(traceback.format_exc())
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
                logger.info(f"Cancelled job {job_id}")
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
