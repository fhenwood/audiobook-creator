"""
Job Worker - Polls database for pending jobs and processes them.

This module provides a worker that can run as part of the main process
or as a standalone service, picking up PENDING jobs from the database.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Awaitable

from audiobook.utils.job_manager import job_manager, Job, JobStatus, JobCheckpoint
from audiobook.utils.database import db
from audiobook.utils.logging_config import get_logger
from audiobook.tts.pipeline import AudiobookPipeline, PipelineConfig

logger = get_logger(__name__)


class JobWorker:
    """
    Worker that polls the database for pending jobs and processes them.
    
    Designed to:
    - Run continuously, polling for new jobs
    - Process one job at a time (GPU constraint)
    - Update job status in DB throughout processing
    - Handle crashes gracefully (jobs become STALLED, can be resumed)
    
    Usage:
        worker = JobWorker()
        await worker.run()  # Runs forever
        
        # Or run as standalone:
        # python -m audiobook.worker
    """
    
    def __init__(
        self,
        poll_interval: int = 5,
        on_job_complete: Optional[Callable[[str], Awaitable[None]]] = None,
        on_job_failed: Optional[Callable[[str, str], Awaitable[None]]] = None
    ):
        """
        Initialize the job worker.
        
        Args:
            poll_interval: Seconds between DB polls when idle.
            on_job_complete: Optional callback when job completes.
            on_job_failed: Optional callback when job fails.
        """
        self.poll_interval = poll_interval
        self.on_job_complete = on_job_complete
        self.on_job_failed = on_job_failed
        self._running = False
        self._current_job_id: Optional[str] = None
    
    async def run(self):
        """Run the worker loop indefinitely."""
        self._running = True
        logger.info("🔄 JobWorker started, polling for jobs...")
        
        while self._running:
            try:
                job = self._claim_next_pending_job()
                
                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    def stop(self):
        """Stop the worker loop."""
        self._running = False
        logger.info("🛑 JobWorker stopping...")
    
    @property
    def current_job(self) -> Optional[str]:
        """Get the currently processing job ID, if any."""
        return self._current_job_id
    
    def _claim_next_pending_job(self) -> Optional[Job]:
        """
        Atomically claim the next pending job.
        
        Returns:
            The claimed Job, or None if no pending jobs.
        """
        try:
            with db.get_cursor() as cursor:
                # Find oldest pending job
                cursor.execute(
                    """
                    SELECT id, data FROM jobs 
                    WHERE status = ? 
                    ORDER BY created_at ASC 
                    LIMIT 1
                    """,
                    (JobStatus.PENDING.value,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                job_id = row['id']
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                # Claim it by setting status to IN_PROGRESS
                now = datetime.now().isoformat()
                job.status = JobStatus.IN_PROGRESS.value
                job.last_activity = now
                job.updated_at = now
                
                cursor.execute(
                    """
                    UPDATE jobs 
                    SET status = ?, updated_at = ?, data = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        JobStatus.IN_PROGRESS.value,
                        now,
                        json.dumps(job.to_dict()),
                        job_id,
                        JobStatus.PENDING.value  # Only claim if still PENDING
                    )
                )
                
                # Check if we actually updated (another worker might have claimed it)
                if cursor.rowcount == 0:
                    return None
                
                logger.info(f"🔒 Claimed job {job_id}: {job.book_title}")
                return job
                
        except Exception as e:
            logger.error(f"Error claiming job: {e}")
            return None
    
    async def _process_job(self, job: Job):
        """Process a claimed job."""
        self._current_job_id = job.job_id
        
        try:
            logger.info(f"🚀 Processing job {job.job_id}: {job.book_title}")
            
            # Update progress
            job_manager.update_job_progress(
                job.job_id,
                "Starting audiobook generation...",
                0.0
            )
            
            # Create pipeline config from job
            config = PipelineConfig(
                job_id=job.job_id,
                book_path=job.checkpoint.get('book_file_path', '') if job.checkpoint else '',
                output_format=job.output_format,
                engine=job.tts_engine,
                voice=job.voice,
                add_emotion_tags=job.checkpoint.get('add_emotion_tags', False) if job.checkpoint else False,
                generate_m4b=job.output_format.lower() == 'm4b',
                use_postprocessing=job.postprocess,
                reference_audio_path=job.checkpoint.get('reference_audio_path', '') if job.checkpoint else '',
                vibevoice_temperature=job.vibevoice_temperature,
                vibevoice_top_p=job.vibevoice_top_p,
            )
            
            # Run pipeline
            pipeline = AudiobookPipeline(config)
            async for progress in pipeline.run():
                job_manager.update_job_progress(
                    job.job_id,
                    progress.message,
                    progress.percent_complete
                )
            
            # Mark complete
            output_path = f"generated_audiobooks/{job.book_title}.{job.output_format}"
            job_manager.complete_job(job.job_id, output_path)
            
            logger.info(f"✅ Job {job.job_id} completed: {output_path}")
            
            if self.on_job_complete:
                await self.on_job_complete(job.job_id)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Job {job.job_id} failed: {error_msg}")
            job_manager.fail_job(job.job_id, error_msg)
            
            if self.on_job_failed:
                await self.on_job_failed(job.job_id, error_msg)
                
        finally:
            self._current_job_id = None


# Singleton instance for use by app.py
_worker: Optional[JobWorker] = None


def get_worker() -> JobWorker:
    """Get or create the global worker instance."""
    global _worker
    if _worker is None:
        _worker = JobWorker()
    return _worker


async def start_worker_background():
    """Start the worker in the background (for use by app.py)."""
    worker = get_worker()
    asyncio.create_task(worker.run())
    return worker


# CLI entry point
if __name__ == "__main__":
    import sys
    
    print("🔧 Starting Audiobook Job Worker...")
    print("Press Ctrl+C to stop")
    
    worker = JobWorker()
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.stop()
        print("\n👋 Worker stopped")
        sys.exit(0)
