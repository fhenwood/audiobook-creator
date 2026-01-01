"""
Job Service - Unified service for job execution.

This module provides a single entry point for audiobook generation jobs
that both the API and Gradio UI use. This ensures consistent behavior
across all clients.
"""

import os
import shutil
import traceback
from typing import Optional, Callable, Awaitable

from audiobook.tts.pipeline import AudiobookPipeline, PipelineConfig
from audiobook.utils.job_manager import job_manager, Job, JobStatus, JobCheckpoint, JobProgress
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobService:
    """
    Unified service for managing audiobook generation jobs.
    
    This is the SINGLE ENTRY POINT for job execution - both API and
    Gradio UI should use this service.
    
    Usage:
        service = JobService()
        
        # Create and start a job
        job = await service.create_and_run_job(
            book_title="My Book",
            book_path="/path/to/book.epub",
            engine="orpheus",
            voice="zac"
        )
        
        # Resume a stalled job
        await service.resume_job(job_id)
    """
    
    def __init__(self):
        self._running_job_id: Optional[str] = None
    
    async def create_and_run_job(
        self,
        book_title: str,
        book_path: str,
        engine: str = "orpheus",
        voice: str = "zac",
        output_format: str = "m4b",
        add_emotion_tags: bool = False,
        dialogue_voice: Optional[str] = None,
        use_postprocessing: bool = False,
        reference_audio_path: Optional[str] = None,
        vibevoice_temperature: float = 0.7,
        vibevoice_top_p: float = 0.95,
        use_vibevoice_dialogue: bool = False,
        vibevoice_dialogue_voice: Optional[str] = None,
        verification_enabled: bool = False,
        on_progress: Optional[Callable[[JobProgress], Awaitable[None]]] = None,
    ) -> Job:
        """
        Create a new job and run it.
        
        Args:
            book_title: Title of the book.
            book_path: Path to the book file.
            engine: TTS engine to use.
            voice: Voice for narration.
            output_format: Output format (m4b, mp3, etc.).
            add_emotion_tags: Whether to use emotion tags.
            dialogue_voice: Optional separate voice for dialogue.
            use_postprocessing: Whether to enhance audio.
            reference_audio_path: For voice cloning (Chatterbox).
            vibevoice_temperature: Temperature for VibeVoice.
            vibevoice_top_p: Top-p for VibeVoice.
            use_vibevoice_dialogue: Use separate dialogue voice (VibeVoice).
            vibevoice_dialogue_voice: Dialogue voice (VibeVoice).
            on_progress: Optional callback for progress updates.
            
        Returns:
            The created Job object.
        """
        # Create job in database
        job = job_manager.create_job(
            book_title=book_title,
            tts_engine=engine,
            voice=voice,
            output_format=output_format,
            postprocess=use_postprocessing,
            vibevoice_voice=vibevoice_dialogue_voice,
            vibevoice_temperature=vibevoice_temperature,
            vibevoice_top_p=vibevoice_top_p,
            use_vibevoice_dialogue=use_vibevoice_dialogue,
            vibevoice_dialogue_voice=vibevoice_dialogue_voice,
            verification_enabled=verification_enabled
        )
        
        job_id = job.job_id
        logger.info(f"Created job {job_id}: {book_title}")
        
        # Copy book text to job directory
        await self._setup_job_files(job_id, book_path)
        
        # Run the job
        await self._run_job(
            job_id=job_id,
            book_path=book_path,
            engine=engine,
            voice=voice,
            output_format=output_format,
            add_emotion_tags=add_emotion_tags,
            dialogue_voice=dialogue_voice,
            use_postprocessing=use_postprocessing,
            reference_audio_path=reference_audio_path,
            vibevoice_temperature=vibevoice_temperature,
            vibevoice_top_p=vibevoice_top_p,
            verification_enabled=verification_enabled,
            on_progress=on_progress,
        )
        
        return job_manager.get_job(job_id)
    
    async def resume_job(
        self,
        job_id: str,
        on_progress: Optional[Callable[[JobProgress], Awaitable[None]]] = None,
    ) -> Job:
        """
        Resume a stalled job from its checkpoint.
        
        Args:
            job_id: ID of the job to resume.
            on_progress: Optional callback for progress updates.
            
        Returns:
            The resumed Job object.
            
        Raises:
            ValueError: If job not found or not resumable.
        """
        job = job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status != JobStatus.STALLED.value:
            raise ValueError(f"Job {job_id} is not stalled (status: {job.status})")
        
        checkpoint = job.get_checkpoint()
        if not checkpoint:
            raise ValueError(f"Job {job_id} has no checkpoint")
        
        # Prepare for resume
        job_manager.prepare_job_for_resume(job_id)
        
        # Run the job from checkpoint
        await self._run_job(
            job_id=job_id,
            book_path=checkpoint.book_file_path,
            engine=job.tts_engine,
            voice=job.voice,
            output_format=job.output_format,
            add_emotion_tags=checkpoint.add_emotion_tags,
            dialogue_voice=None,  # TODO: Store in checkpoint
            use_postprocessing=job.postprocess,
            reference_audio_path=checkpoint.reference_audio_path,
            vibevoice_temperature=job.vibevoice_temperature,
            vibevoice_top_p=job.vibevoice_top_p,
            verification_enabled=job.verification_enabled if hasattr(job, 'verification_enabled') else False,
            on_progress=on_progress,
        )
        
        return job_manager.get_job(job_id)
    
    async def _run_job(
        self,
        job_id: str,
        book_path: str,
        engine: str,
        voice: str,
        output_format: str,
        add_emotion_tags: bool,
        dialogue_voice: Optional[str],
        use_postprocessing: bool,
        reference_audio_path: Optional[str],
        vibevoice_temperature: float,
        vibevoice_top_p: float,
        verification_enabled: bool = False,
        on_progress: Optional[Callable[[JobProgress], Awaitable[None]]] = None,
    ):
        """Run the audiobook generation pipeline."""
        self._running_job_id = job_id
        
        try:
            # Normalize output format
            fmt = output_format.lower()
            if "m4b" in fmt:
                fmt = "m4b"
            
            # Create pipeline config
            config = PipelineConfig(
                job_id=job_id,
                book_path=book_path,
                output_format=fmt,
                engine=engine,
                voice=voice,
                dialogue_voice=dialogue_voice,
                use_separate_dialogue=bool(dialogue_voice),
                add_emotion_tags=add_emotion_tags,
                generate_m4b=(fmt == "m4b"),
                use_postprocessing=use_postprocessing,
                reference_audio_path=reference_audio_path,
                vibevoice_temperature=vibevoice_temperature,

                vibevoice_top_p=vibevoice_top_p,
                verification_enabled=verification_enabled,
            )
            
            # Run pipeline
            pipeline = AudiobookPipeline(config)
            async for progress in pipeline.run():
                # Update job progress
                job_manager.update_job_progress(
                    job_id,
                    progress.message,
                    progress.percent_complete
                )
                
                # Call progress callback if provided
                if on_progress:
                    await on_progress(progress)
            
            # Mark complete
            output_path = f"generated_audiobooks/audiobook.{fmt}"
            job_manager.complete_job(job_id, output_path)
            logger.info(f"✅ Job {job_id} completed: {output_path}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Job {job_id} failed: {error_msg}")
            logger.error(traceback.format_exc())
            job_manager.fail_job(job_id, error_msg)
            raise
        finally:
            self._running_job_id = None
    
    async def _setup_job_files(self, job_id: str, book_path: str):
        """Copy necessary files to job directory."""
        # Ensure job directory exists
        job_manager.create_job_directory(job_id)
        
        # Copy converted book text to job directory if it exists
        if os.path.exists("converted_book.txt"):
            job_converted_path = job_manager.get_job_converted_book_path(job_id)
            shutil.copy2("converted_book.txt", job_converted_path)
            logger.info(f"Copied converted book to job directory")
        
        # Copy emotion tags if they exist
        if os.path.exists("tag_added_lines_chunks.txt"):
            job_emotion_path = job_manager.get_job_emotion_tags_path(job_id)
            shutil.copy2("tag_added_lines_chunks.txt", job_emotion_path)
            logger.info(f"Copied emotion tags to job directory")
    
    def is_job_running(self) -> bool:
        """Check if a job is currently running."""
        return self._running_job_id is not None
    
    @property
    def current_job_id(self) -> Optional[str]:
        """Get the currently running job ID."""
        return self._running_job_id


# Global service instance
job_service = JobService()
