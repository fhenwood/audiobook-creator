"""
Background job runners for Audiobook Creator.

Contains async functions that run audiobook generation jobs independently of Gradio.
Extracted from app.py for modularity.
"""

import os
import shutil
import traceback
from audiobook.core.job_service import job_service
from audiobook.utils.job_manager import job_manager, JobProgress
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


async def run_audiobook_job_background(
    job_id: str,
    tts_engine: str, 
    narrator_voice: str, 
    output_format: str, 
    book_file_path: str,
    book_title: str,
    add_emotion_tags: bool,
    dialogue_voice: str = None,
    reference_audio_path: str = None,
    postprocess: bool = False,
    vibevoice_voice: str = None,
    vibevoice_temperature: float = 0.7,
    vibevoice_top_p: float = 0.95,
    use_vibevoice_dialogue: bool = False,
    vibevoice_dialogue_voice: str = None,
    verification_enabled: bool = False
):
    """
    Background async function that runs the audiobook generation.
    This runs independently of the Gradio connection.
    Progress is saved to job_manager so users can check status even after closing browser.
    """
    try:
        # Use the unified JobService to run the job
        # This uses the new AudiobookPipeline architecture
        await job_service._run_job(
            job_id=job_id,
            book_path=book_file_path,
            engine=tts_engine,
            voice=narrator_voice,
            output_format=output_format,
            add_emotion_tags=add_emotion_tags,
            dialogue_voice=dialogue_voice,
            use_postprocessing=postprocess,
            reference_audio_path=reference_audio_path,
            vibevoice_temperature=vibevoice_temperature,
            vibevoice_top_p=vibevoice_top_p,
            verification_enabled=verification_enabled if 'verification_enabled' in locals() else False, # Parameter verification_enabled IS passed in signature
            # Note: We need to ensure verification_enabled is in the function signature in app/jobs.py if it isn't already. 
            # I added it in previous steps? No, I checked app/jobs.py and verification_enabled was NOT in the signature in the view_file output.
            # I need to update the signature too.
        )

    except Exception as e:
        logger.error(f"❌ Job {job_id}: Error - {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))


async def resume_job_background(job_id: str, job, checkpoint):
    """
    Background async function that resumes a stalled audiobook generation job.
    This runs independently of the Gradio connection.
    """
    try:
        # Use JobService to resume
        await job_service.resume_job(job_id)

    except Exception as e:
        logger.error(f"❌ Job {job_id}: Resume error - {e}", exc_info=True)
        job_manager.fail_job(job_id, str(e))
