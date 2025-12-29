"""
Audiobook Creator

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import gradio as gr
import os
import shutil
import traceback
import tempfile
from fastapi import FastAPI
from openai import OpenAI
from audiobook.core.text_extraction import process_book_and_extract_text, save_book
from audiobook.tts.generator import process_audiobook_generation, validate_book_for_m4b_generation, sanitize_filename
from audiobook.core.emotion_tags import process_emotion_tags
from audiobook.tts.voice_mapping import get_available_voices, get_voice_list
from audiobook.tts.service import tts_service
from audiobook.core.voice_manager import voice_manager
from audiobook.utils.job_manager import job_manager, JobStatus, get_jobs_dataframe, auto_resume_service
from audiobook.utils.gpu_resource_manager import gpu_manager
from audiobook.utils.background_runner import background_runner
from audiobook.models.manager import model_manager, ModelType
from dotenv import load_dotenv

# Modular imports from app package
from app.voice_utils import (
    extract_title_from_filename,
    get_vibevoice_choices,
    get_voice_choices,
    get_installed_tts_engines,
    check_llm_availability,
)
from app.handlers import (
    validate_book_upload,
    text_extraction_wrapper,
    save_book_wrapper,
    generate_voice_sample,
)
from app.jobs import (
    run_audiobook_job_background as _run_audiobook_job_background,
    resume_job_background as _resume_job_background,
)


load_dotenv()

# TTS Configuration
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1")
TTS_API_KEY = os.environ.get("TTS_API_KEY", "not-needed")

# Create voice samples directory
os.makedirs("static_files/voice_samples", exist_ok=True)

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
.voice-card {border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 8px;}
"""

app = FastAPI()

# Mount REST API endpoints for backend/frontend separation
# This enables CLI, React, iOS, etc. frontends to access the TTS engine
from audiobook.api import api_app
app.mount("/api", api_app)

# Auto-resume callback - This will be set up after the Gradio app is defined
# For now, we just mark stalled jobs on startup and let users manually resume
def startup_stall_check():
    """Check for jobs that were running when the server stopped and mark them as stalled."""
    stalled = auto_resume_service.check_startup_stalled_jobs()
    if stalled:
        print(f"🔄 Found {len(stalled)} job(s) that were in-progress and are now marked as stalled")
        print(f"   Job IDs: {stalled}")
        print(f"   These jobs can be resumed from the Jobs Dashboard")
    
    # Start the auto-resume monitoring service
    auto_resume_service.start()

# Run startup check
startup_stall_check()

def extract_title_from_filename(book_file):
    """Extract book title from uploaded filename without extension"""
    if book_file is None:
        return ""
    filename = os.path.basename(book_file.name)
    title = os.path.splitext(filename)[0]
    return sanitize_filename(title)

def validate_book_upload(book_file, book_title):
    """Validate book upload and return a notification"""
    if book_file is None:
        return gr.Warning("Please upload a book file first.")
    
    if not book_title:
        book_title = os.path.splitext(os.path.basename(book_file.name))[0]

    book_title = sanitize_filename(book_title)
    
    yield book_title
    return gr.Info(f"Book '{book_title}' ready for processing.", duration=5)

def text_extraction_wrapper(book_file):
    """Wrapper for text extraction with validation and progress updates"""
    if book_file is None:
        yield None
        return gr.Warning("Please upload a book file and enter a title first.")
    
    try:
        last_output = None
        # Pass through all yield values from the original function (always using calibre)
        for output in process_book_and_extract_text(book_file):
            last_output = output
            yield output  # Yield each progress update
        
        # Final yield with success notification
        yield last_output
        return gr.Info("Text extracted successfully! You can now edit the content.", duration=5)
    except ValueError as e:
        # Handle validation errors specifically
        print(e)
        traceback.print_exc()
        yield None
        return gr.Warning(f"Book validation error: {str(e)}")
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield None
        return gr.Warning(f"Error extracting text: {str(e)}")

def save_book_wrapper(text_content):
    """Wrapper for saving book with validation"""
    if not text_content:
        return gr.Warning("No text content to save.")
    
    try:
        save_book(text_content)
        return gr.Info("📖 Book saved successfully as 'converted_book.txt'!", duration=10)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.Warning(f"Error saving book: {str(e)}")

async def _run_audiobook_job_background(
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
    vibevoice_dialogue_voice: str = None
):
    """
    Background async function that runs the audiobook generation.
    This runs independently of the Gradio connection.
    Progress is saved to job_manager so users can check status even after closing browser.
    """
    need_orpheus_tts = tts_engine == "Orpheus"
    
    try:
        # =====================================================================
        # PHASE 1: Emotion Tags (if enabled) - Uses LLM server (~12GB VRAM)
        # =====================================================================
        if add_emotion_tags:
            job_manager.update_job_progress(job_id, "🚀 Starting LLM server for emotion tags...")
            
            # Acquire LLM server
            success, gpu_message = gpu_manager.acquire_llm()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start LLM server: {gpu_message}")
                return
            
            try:
                job_manager.update_job_progress(job_id, "🎭 Processing emotion tags...")
                
                # Run emotion tags processing
                async for progress in process_emotion_tags(False):
                    job_manager.update_job_progress(job_id, f"🎭 Emotion tags: {str(progress)[:100]}")
                
                # Copy emotion tags file to job directory for persistence
                if os.path.exists("tag_added_lines_chunks.txt"):
                    job_emotion_tags_path = job_manager.get_job_emotion_tags_path(job_id)
                    shutil.copy2("tag_added_lines_chunks.txt", job_emotion_tags_path)
                
                job_manager.update_job_progress(job_id, "✅ Emotion tags added successfully!")
            except Exception as e:
                job_manager.fail_job(job_id, f"Emotion tags failed: {str(e)}")
                traceback.print_exc()
                return
            finally:
                # IMPORTANT: Release LLM server BEFORE starting TTS to free ~12GB VRAM
                print(f"🔄 Job {job_id}: Releasing LLM server to free VRAM for TTS...")
                gpu_manager.release_llm(immediate=True)
        
        # =====================================================================
        # PHASE 2: TTS Generation - Uses Orpheus LLM (~5GB VRAM) for Orpheus engine
        # =====================================================================
        if need_orpheus_tts:
            job_manager.update_job_progress(job_id, "🚀 Starting Orpheus LLM for TTS...")
            
            # Acquire Orpheus LLM
            success, gpu_message = gpu_manager.acquire_orpheus()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start Orpheus LLM: {gpu_message}")
                return
        
        try:
            job_manager.update_job_progress(job_id, "Starting audiobook generation...")
            
            # Pass through all yield values from the original function
            async for output in process_audiobook_generation(
                "Single Voice", 
                narrator_voice, 
                output_format, 
                book_file_path, 
                add_emotion_tags,
                tts_engine=tts_engine,
                reference_audio_path=reference_audio_path,
                job_id=job_id,
                dialogue_voice=dialogue_voice,
                use_postprocessing=postprocess,
                vibevoice_voice=vibevoice_voice,
                vibevoice_temperature=vibevoice_temperature,
                vibevoice_top_p=vibevoice_top_p,
                use_vibevoice_dialogue=use_vibevoice_dialogue,
                vibevoice_dialogue_voice=vibevoice_dialogue_voice
            ):
                # Update job progress with current output
                if output:
                    job_manager.update_job_progress(job_id, str(output)[:200])
            
            # Get the correct file extension based on the output format
            generate_m4b_audiobook_file = output_format == "M4B (Chapters & Cover)"
            file_extension = "m4b" if generate_m4b_audiobook_file else output_format.lower()
            
            # Set the audiobook file path according to the provided information
            audiobook_path = os.path.join("generated_audiobooks", f"audiobook.{file_extension}")

            # Rename the audiobook file to the book title
            final_path = os.path.join("generated_audiobooks", f"{book_title}.{file_extension}")
            if os.path.exists(audiobook_path):
                os.rename(audiobook_path, final_path)
                audiobook_path = final_path
            
            # Mark job as completed
            job_manager.complete_job(job_id, audiobook_path)
            print(f"✅ Job {job_id}: Audiobook generated successfully!")
            
        finally:
            # Release Orpheus LLM after TTS generation
            if need_orpheus_tts:
                print(f"🔓 Job {job_id}: Releasing Orpheus LLM")
                gpu_manager.release_orpheus()
                
    except Exception as e:
        print(f"❌ Job {job_id}: Error - {e}")
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))


async def generate_audiobook_wrapper(tts_engine, narrator_voice, output_format, book_file, add_emotion_tags_checkbox, book_title=None, use_dialogue_voice_checkbox=False, dialogue_voice_selection=None, reference_audio=None, vibevoice_voice=None, postprocess=False, vibevoice_temperature=0.7, vibevoice_top_p=0.95, use_vibevoice_dialogue=False, vibevoice_dialogue_voice=None):
    """Wrapper for audiobook generation - starts a BACKGROUND JOB that runs independently.
    
    The job continues running even if you close the browser tab. Check the Jobs tab for progress.
    
    GPU Memory Optimization:
    - Phase 1 (Emotion Tags): Load LLM server (~12GB), process tags, then UNLOAD
    - Phase 2 (TTS): Load Orpheus LLM (~5GB), generate audio, then unload
    - This ensures we never need both models in VRAM at once!
    
    Args:
        use_dialogue_voice_checkbox: If True, use separate voice for dialogue
        dialogue_voice_selection: The voice to use for dialogue (text in quotes)
        vibevoice_voice: Selected speaker for VibeVoice engine
    """
    if book_file is None:
        yield gr.Warning("Please upload a book file first."), None, None
        yield None, None, None
        return
    if not output_format:
        yield gr.Warning("Please select an output format."), None, None
        yield None, None, None
        return
    
    # If VibeVoice engine, use the VibeVoice selector value as narrator
    if tts_engine == "VibeVoice" and vibevoice_voice:
        narrator_voice = vibevoice_voice
    
    # Validate Chatterbox requirements
    if tts_engine == "Chatterbox":
        if reference_audio is None or not os.path.exists(reference_audio):
            yield gr.Warning("Please upload a reference audio sample for Chatterbox voice cloning."), None, None
            yield None, None, None
            return
        voice_display = "cloned voice"
    elif tts_engine == "VibeVoice":
        voice_display = vibevoice_voice if vibevoice_voice else "speaker_0"
    else:
        voice_display = narrator_voice
        if use_dialogue_voice_checkbox and dialogue_voice_selection:
            voice_display = f"{narrator_voice} (dialogue: {dialogue_voice_selection})"
    
    # Create job for tracking
    job = job_manager.create_job(
        book_title=book_title or "Untitled",
        tts_engine=tts_engine,
        voice=voice_display,
        output_format=output_format,
        postprocess=postprocess,
        vibevoice_voice=vibevoice_voice,
        vibevoice_temperature=vibevoice_temperature,
        vibevoice_top_p=vibevoice_top_p,
        use_vibevoice_dialogue=use_vibevoice_dialogue,
        vibevoice_dialogue_voice=vibevoice_dialogue_voice
    )
    job_id = job.job_id
    
    # Determine what we need
    add_emotion_tags = add_emotion_tags_checkbox if tts_engine == "Orpheus" else False
    
    # Early validation for M4B format
    if output_format == "M4B (Chapters & Cover)":
        job_manager.update_job_progress(job_id, "Validating book file...")
        yield gr.Info(f"Job {job_id}: Validating book file for M4B audiobook generation..."), None, job_id
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_file)
        
        if not is_valid:
            job_manager.fail_job(job_id, f"Book validation failed: {error_message}")
            yield gr.Warning(f"❌ Job {job_id}: Book validation failed: {error_message}"), None, job_id
            yield None, None, job_id
            return
            
        yield gr.Info(f"✅ Job {job_id}: Book validation successful! Title: {metadata.get('Title', 'Unknown')}"), None, job_id
    
    # Copy converted_book.txt to job directory for persistence
    if os.path.exists("converted_book.txt"):
        job_converted_book_path = job_manager.get_job_converted_book_path(job_id)
        shutil.copy2("converted_book.txt", job_converted_book_path)
    
    # Determine dialogue voice (None means use narrator voice for everything)
    effective_dialogue_voice = None
    if use_dialogue_voice_checkbox and dialogue_voice_selection:
        effective_dialogue_voice = dialogue_voice_selection
    
    # Store book file path for background processing
    book_file_path = book_file.name if hasattr(book_file, 'name') else book_file
    
    # Submit job to background runner - THIS RUNS INDEPENDENTLY OF GRADIO
    def create_job_coro():
        return _run_audiobook_job_background(
            job_id=job_id,
            tts_engine=tts_engine,
            narrator_voice=narrator_voice,
            output_format=output_format,
            book_file_path=book_file_path,
            book_title=book_title or "Untitled",
            add_emotion_tags=add_emotion_tags,
            dialogue_voice=effective_dialogue_voice,
            reference_audio_path=reference_audio,
            postprocess=postprocess,
            vibevoice_voice=vibevoice_voice,
            vibevoice_temperature=vibevoice_temperature,
            vibevoice_top_p=vibevoice_top_p,
            use_vibevoice_dialogue=use_vibevoice_dialogue,
            vibevoice_dialogue_voice=vibevoice_dialogue_voice
        )
    
    if background_runner.submit_job(job_id, create_job_coro):
        yield gr.Info(f"🚀 Job {job_id} started in background! You can safely close this tab - check Jobs tab for progress.", duration=10), None, job_id
        yield f"🎙️ Job {job_id} is now running in the background.\n\n👉 **You can close this browser tab** - the job will continue running.\n\n📋 Check the **Jobs** tab to monitor progress and download when complete.", None, job_id
    else:
        # A job is already running
        running_jobs = background_runner.get_running_jobs()
        job_manager.fail_job(job_id, "Another job is already running")
        yield gr.Warning(f"❌ Cannot start job: Another job ({running_jobs[0] if running_jobs else 'unknown'}) is already running. Wait for it to complete."), None, job_id
        yield None, None, job_id
        return


async def _resume_job_background(job_id: str, job, checkpoint):
    """
    Background async function that resumes a stalled audiobook generation job.
    This runs independently of the Gradio connection.
    """
    need_orpheus_tts = job.tts_engine == "Orpheus"
    add_emotion_tags = checkpoint.add_emotion_tags
    
    # Get job-specific paths
    job_converted_book_path = job_manager.get_job_converted_book_path(job_id)
    job_emotion_tags_path = job_manager.get_job_emotion_tags_path(job_id)
    
    try:
        # Check if emotion tags need to be re-processed
        emotion_tags_exist = os.path.exists(job_emotion_tags_path) or os.path.exists("tag_added_lines_chunks.txt")
        if add_emotion_tags and not emotion_tags_exist:
            job_manager.update_job_progress(job_id, "🎭 Re-processing emotion tags...")
            
            # First make sure converted_book.txt exists
            converted_book_exists = os.path.exists(job_converted_book_path) or os.path.exists("converted_book.txt")
            if not converted_book_exists:
                book_path = checkpoint.book_file_path
                if not os.path.exists(book_path):
                    book_filename = os.path.basename(book_path)
                    persistent_path = os.path.join("generated_audiobooks", f"_temp_{book_filename}")
                    if os.path.exists(persistent_path):
                        book_path = persistent_path
                    else:
                        job_manager.fail_job(job_id, "Book file not found for re-extracting text")
                        return
                
                # Extract text from book
                import subprocess
                result = subprocess.run(
                    ["ebook-convert", book_path, job_converted_book_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    job_manager.fail_job(job_id, f"Failed to extract text: {result.stderr}")
                    return
                
                shutil.copy2(job_converted_book_path, "converted_book.txt")
            
            # Re-process emotion tags
            success, gpu_message = gpu_manager.acquire_llm()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start LLM server: {gpu_message}")
                return
            
            try:
                async for progress in process_emotion_tags(False):
                    job_manager.update_job_progress(job_id, f"🎭 Emotion tags: {str(progress)[:100]}")
                
                if os.path.exists("tag_added_lines_chunks.txt"):
                    shutil.copy2("tag_added_lines_chunks.txt", job_emotion_tags_path)
            except Exception as e:
                job_manager.fail_job(job_id, f"Emotion tags failed: {str(e)}")
                traceback.print_exc()
                return
            finally:
                gpu_manager.release_llm(immediate=True)
        
        # Acquire Orpheus LLM if needed
        if need_orpheus_tts:
            job_manager.update_job_progress(job_id, "🚀 Starting Orpheus LLM for resume...")
            success, gpu_message = gpu_manager.acquire_orpheus()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start Orpheus LLM: {gpu_message}")
                return
        
        try:
            job_manager.update_job_progress(job_id, f"Resuming from {checkpoint.lines_completed} lines...")
            
            # Parse voice string to extract narrator and dialogue voices
            narrator_voice = job.voice
            dialogue_voice = None
            if " (dialogue: " in job.voice:
                parts = job.voice.split(" (dialogue: ")
                narrator_voice = parts[0]
                dialogue_voice = parts[1].rstrip(")")
                print(f"📢 Parsed resume voices: narrator={narrator_voice}, dialogue={dialogue_voice}")
            
            # Resume generation with checkpoint
            async for output in process_audiobook_generation(
                "Single Voice", 
                narrator_voice, 
                job.output_format, 
                checkpoint.book_file_path, 
                checkpoint.add_emotion_tags,
                tts_engine=job.tts_engine,
                reference_audio_path=checkpoint.reference_audio_path if hasattr(checkpoint, 'reference_audio_path') else None,
                job_id=job_id,
                resume_checkpoint=checkpoint.to_dict(),
                dialogue_voice=dialogue_voice,
                use_postprocessing=job.postprocess if hasattr(job, 'postprocess') else False,
                vibevoice_voice=job.vibevoice_voice if hasattr(job, 'vibevoice_voice') else None,
                vibevoice_temperature=job.vibevoice_temperature if hasattr(job, 'vibevoice_temperature') else 0.7,
                vibevoice_top_p=job.vibevoice_top_p if hasattr(job, 'vibevoice_top_p') else 0.95,
                use_vibevoice_dialogue=job.use_vibevoice_dialogue if hasattr(job, 'use_vibevoice_dialogue') else False,
                vibevoice_dialogue_voice=job.vibevoice_dialogue_voice if hasattr(job, 'vibevoice_dialogue_voice') else None
            ):
                if output:
                    job_manager.update_job_progress(job_id, str(output)[:200])
            
            # Get the correct file extension
            generate_m4b_audiobook_file = job.output_format == "M4B (Chapters & Cover)"
            file_extension = "m4b" if generate_m4b_audiobook_file else job.output_format.lower()
            
            # Set the audiobook file path
            audiobook_path = os.path.join("generated_audiobooks", f"audiobook.{file_extension}")
            
            # Rename to book title
            final_path = os.path.join("generated_audiobooks", f"{job.book_title}.{file_extension}")
            if os.path.exists(audiobook_path):
                os.rename(audiobook_path, final_path)
                audiobook_path = final_path
            
            # Mark job as completed
            job_manager.complete_job(job_id, audiobook_path)
            print(f"✅ Job {job_id}: Resume completed successfully!")
            
        finally:
            if need_orpheus_tts:
                print(f"🔓 Job {job_id}: Releasing Orpheus LLM")
                gpu_manager.release_orpheus()
                
    except Exception as e:
        print(f"❌ Job {job_id}: Resume error - {e}")
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))


async def resume_job_wrapper(job_id):
    """Resume a stalled job from its checkpoint - runs in BACKGROUND.
    
    The job continues running even if you close the browser tab. Check the Jobs tab for progress.
    """
    if not job_id:
        yield gr.Warning("Please enter a Job ID to resume"), None, job_id
        yield None, None, job_id
        return
    
    job = job_manager.get_job(job_id)
    if not job:
        yield gr.Warning(f"Job {job_id} not found"), None, job_id
        yield None, None, job_id
        return
    
    if job.status != JobStatus.STALLED.value:
        yield gr.Warning(f"Job {job_id} is not in a resumable state (status: {job.status})"), None, job_id
        yield None, None, job_id
        return
    
    checkpoint = job.get_checkpoint()
    if not checkpoint:
        yield gr.Warning(f"Job {job_id} has no checkpoint data to resume from"), None, job_id
        yield None, None, job_id
        return
    
    # Prepare job for resume
    job_manager.prepare_job_for_resume(job_id)
    
    # Ensure job directory exists
    job_manager.create_job_directory(job_id)
    
    # Submit to background runner
    def create_resume_coro():
        return _resume_job_background(job_id, job, checkpoint)
    
    if background_runner.submit_job(job_id, create_resume_coro):
        yield gr.Info(f"🔄 Job {job_id} resumed in background! You can safely close this tab.", duration=10), None, job_id
        yield f"🎙️ Job {job_id} is continuing in the background.\n\n👉 **You can close this browser tab** - the job will continue running.\n\n📋 Check the **Jobs** tab to monitor progress and download when complete.", None, job_id
    else:
        # A job is already running
        running_jobs = background_runner.get_running_jobs()
        job_manager.fail_job(job_id, "Another job is already running")
        yield gr.Warning(f"❌ Cannot resume job: Another job ({running_jobs[0] if running_jobs else 'unknown'}) is already running."), None, job_id
        yield None, None, job_id
        return

async def generate_voice_sample(engine_id, voice_id, sample_text, reference_audio=None, vibevoice_voice=None, use_postprocessing=False, vibevoice_temperature=0.7, vibevoice_top_p=0.95):
    """Generate a voice sample using the selected engine."""
    if not sample_text or not sample_text.strip():
        return None, "Please enter some text to generate a sample."
    
    try:
        engine_id = engine_id.lower()
        
        # Determine voice ID based on engine
        if engine_id == "orpheus":
            if not voice_id: return None, "Please select an Orpheus voice."
        elif engine_id == "vibevoice":
            # Override voice_id from Orpheus selector with VibeVoice selector input
            voice_id = vibevoice_voice
            if not voice_id: return None, "Please select a VibeVoice speaker."
            
        elif engine_id == "chatterbox":
            if reference_audio is None: return None, "Please upload reference audio."
        
        # Handle zero-shot via voice selection (if voice_id is a path)
        # For VibeVoice: keep the path as voice_id (engine checks voice param for file paths)
        # For Chatterbox: use reference_audio parameter
        if voice_id and os.path.exists(voice_id) and engine_id == "chatterbox":
            # Chatterbox uses reference_audio for cloning
            reference_audio = voice_id
            voice_id = "cloned"
        # For VibeVoice, keep the file path in voice_id (the engine handles it)
            
        # Generate using unified service
        # Note: tts_service handles GPU resource acquisition automatically
        result = await tts_service.generate(
            text=sample_text.strip(),
            engine=engine_id,
            voice=voice_id,
            reference_audio=reference_audio,
            speed=0.9 if engine_id == "orpheus" else 1.0,
            temperature=vibevoice_temperature if engine_id == "vibevoice" else 0.7,
            top_p=vibevoice_top_p if engine_id == "vibevoice" else 0.95
        )
        
        # Save to file - use basename if voice_id is a path
        voice_name = voice_id
        if voice_id and isinstance(voice_id, str) and ('/' in voice_id or '\\' in voice_id):
            voice_name = os.path.splitext(os.path.basename(voice_id))[0]
            
        filename = f"{engine_id}_{voice_name or 'cloned'}_sample.wav"
        filepath = os.path.join("static_files/voice_samples", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            f.write(result.audio_data)
            
        status = f"✅ Generated sample with {engine_id}"
        
        # Apply Enhanced Post-processing if requested
        if use_postprocessing:
            try:
                from audiobook.utils.audio_enhancer import audio_pipeline
                enhanced_path = os.path.join("static_files/voice_samples", f"enhanced_{filename}")
                # Run only enhancement stages on TTS output (SR + Enhancement, skip Demucs)
                processed_path, status_msg = audio_pipeline.process(
                    filepath, 
                    output_path=enhanced_path,
                    enable_preprocessing=True,
                    stages=['enhancement', 'sr']
                )
                if processed_path and os.path.exists(processed_path):
                    filepath = processed_path
                    status += f" + {status_msg}"
            except Exception as e:
                print(f"⚠️ Post-processing failed: {e}")
                status += " (Post-processing failed)"
            
        return filepath, status
        
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def get_vibevoice_choices():
    """Get VibeVoice choices from Voice Library only (no generic speakers)."""
    # Only return custom voices from Voice Library (no generic speaker_0-3)
    custom = [(f"🎤 {v.name}", v.path) for v in voice_manager.list_voices()]
    if not custom:
        # Fallback if no voices uploaded yet
        return [("No voices - add some in Voice Library", "")]
    return custom

def get_voice_choices():
    """Get voice choices for dropdown with descriptions."""
    # Standard voices
    voices = get_available_voices()
    choices = [(f"{name} - {desc}", name) for name, desc in voices.items()]
    
    # Custom voices
    custom_voices = voice_manager.list_voices()
    for v in custom_voices:
        choices.append((f"🎤 {v.name} (Custom)", v.path))
        
    return choices

def get_installed_tts_engines():
    """Get list of installed TTS engines."""
    engines = []
    
    # Orpheus is the core engine, assumed available via container services
    # Ideally we'd check checking health, but for now we always show it 
    # if the service is configured.
    engines.append("Orpheus")
    
    # Chatterbox requires XTTS v2 (placeholder assumption) or its own model
    # User feedback indicates XTTS and Orpheus are distinct.
    if model_manager.get_model_path("xtts-v2"):
        engines.append("Chatterbox")
        
    # VibeVoice requires vibevoice-7b
    if model_manager.get_model_path("vibevoice-7b"):
        engines.append("VibeVoice")
        
    if not engines:
        engines.append("(No Engines Installed)")
        
    return engines

def check_llm_availability():
    """Check if any LLM is installed for emotion tagging."""
    models = model_manager.list_models()
    for m in models:
        # Check if type matches LLM and is installed
        if m.definition.type == ModelType.LLM and m.installed:
            return True
    return False

with gr.Blocks(css=css, theme=gr.themes.Default()) as gradio_app:
    gr.Markdown("# 📖 Audiobook Creator")
    gr.Markdown("Create professional audiobooks from your ebooks using Orpheus TTS or Chatterbox (with zero-shot voice cloning).")
    
    # Get voice choices once for use in all tabs
    voice_choices = get_voice_list()
    voice_descriptions = get_available_voices()
    
    with gr.Tabs():
        # ==================== Voice Sampling Tab ====================
        with gr.TabItem("🎙️ Voice Sampling"):
            gr.Markdown("### Preview TTS Voices")
            gr.Markdown("Test different voices with your own text before creating an audiobook.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # TTS Engine selector
                    available_engines = get_installed_tts_engines()
                    tts_engine_sampling = gr.Radio(
                        choices=available_engines,
                        label="TTS Engine",
                        value=available_engines[0] if available_engines else None,
                        info="Orpheus: Emotion control. Chatterbox: Voice cloning. VibeVoice: High fidelity."
                    )
                    
                    # Orpheus voice selector (visible by default)
                    with gr.Group(visible=True) as orpheus_voice_group:
                        voice_selector = gr.Dropdown(
                            choices=voice_choices,
                            label="Select Voice",
                            value="zac",
                            info="Choose an Orpheus voice to preview"
                        )
                        
                        # Show voice description
                        voice_description = gr.Textbox(
                            label="Voice Description",
                            value=voice_descriptions.get("zac", ""),
                            interactive=False
                        )
                    
                    # Chatterbox zero-shot reference audio (hidden by default)
                    with gr.Group(visible=False) as chatterbox_voice_group:
                        gr.Markdown("""🎤 **Zero-Shot Voice Cloning with Chatterbox**
                        
                        Upload a ~10 second reference audio sample to clone any voice.
                        """)
                        reference_audio_sampling = gr.Audio(
                            label="Reference Audio Sample (Required for Chatterbox)",
                            type="filepath",
                            sources=["upload", "microphone"],
                            interactive=True
                        )

                    # VibeVoice selector (hidden by default)
                    with gr.Group(visible=False) as vibevoice_voice_group:
                        gr.Markdown("### VibeVoice Speakers")
                        vibevoice_selector = gr.Dropdown(
                            choices=get_vibevoice_choices(),
                            label="Select Speaker",
                            value="speaker_0",
                            info="Select one of the multi-speaker voices or a custom voice from Voice Library"
                        )
                        
                        with gr.Row():
                            vibevoice_temperature_sampling = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                info="Lower = Stable, Higher = Expressive/Random"
                            )
                            vibevoice_top_p_sampling = gr.Slider(
                                label="Top P",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                info="Lower = Focused, Higher = Diverse"
                            )
                    
                    sample_text = gr.Textbox(
                        label="Sample Text",
                        placeholder="Enter text to hear the voice sample...",
                        value="Hello! This is a sample of my voice. I can read your books with emotion and expression, bringing characters to life.",
                        lines=4,
                        info="Enter any text you want to hear in the selected voice"
                    )
                    
                    sample_postprocess = gr.Checkbox(
                        label="✨ Enhanced Post-processing", 
                        value=False,
                        info="Run output through audio enhancement (Speech Enhancement + Super-Resolution)"
                    )
                    
                    generate_sample_btn = gr.Button("🔊 Generate Sample", variant="primary")
                    
                    sample_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Click 'Generate Sample' to hear the voice"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🎧 Audio Preview")
                    audio_preview = gr.Audio(
                        label="Voice Sample",
                        type="filepath",
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    ### Orpheus Voices
                    
                    **Female Voices:**
                    - **Tara** - Warm, expressive narrator
                    - **Leah** - Soft, gentle tone
                    - **Jess** - Energetic, youthful
                    - **Mia** - Clear, articulate
                    - **Zoe** - Neutral, versatile
                    
                    **Male Voices:**
                    - **Leo** - Deep, authoritative
                    - **Dan** - Friendly, conversational
                    - **Zac** - Strong, confident narrator
                    
                    ---
                    ### Chatterbox
                    
                    Zero-shot voice cloning - upload any ~10 second audio sample to clone that voice!
                    
                    Supports paralinguistic tags: `[laugh]`, `[chuckle]`, `[cough]`, etc.
                    """)
            
            # TTS engine change handler - toggle visibility of voice groups
            def update_tts_engine_visibility(tts_engine):
                if tts_engine == "Orpheus":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
                elif tts_engine == "Chatterbox":
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                else: # VibeVoice
                    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            
            tts_engine_sampling.change(
                update_tts_engine_visibility,
                inputs=[tts_engine_sampling],
                outputs=[orpheus_voice_group, chatterbox_voice_group, vibevoice_voice_group]
            )
            
            # Voice selector change handler
            def update_voice_description(voice_name):
                descriptions = get_available_voices()
                return descriptions.get(voice_name, "No description available")
            
            voice_selector.change(
                update_voice_description,
                inputs=[voice_selector],
                outputs=[voice_description]
            )
            
            # Generate sample button handler
            generate_sample_btn.click(
                generate_voice_sample,
                inputs=[tts_engine_sampling, voice_selector, sample_text, reference_audio_sampling, vibevoice_selector, sample_postprocess, vibevoice_temperature_sampling, vibevoice_top_p_sampling],
                outputs=[audio_preview, sample_status]
            )
        
        # ==================== Audiobook Creation Tab ====================
        with gr.TabItem("📖 Create Audiobook"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('<div class="step-heading">📚 Step 1: Book Details</div>')
                    
                    book_title = gr.Textbox(
                        label="Book Title", 
                        placeholder="Enter the title of your book",
                        info="This will be used for naming the audiobook file"
                    )
                    
                    book_input = gr.File(
                        label="Upload Book"
                    )
                    
                    validate_btn = gr.Button("Validate Book", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown('<div class="step-heading">✂️ Step 2: Extract & Edit Content</div>')
                    
                    convert_btn = gr.Button("Extract Text", variant="primary")
                    
                    with gr.Accordion("Editing Tips", open=True):
                        gr.Markdown("""
                        * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements
                        * Fix formatting issues or OCR errors
                        * Check for chapter breaks and paragraph formatting
                        """)
                    
                    # Navigation buttons for the textbox
                    with gr.Row():
                        top_btn = gr.Button("↑ Go to Top", size="sm", variant="secondary")
                        bottom_btn = gr.Button("↓ Go to Bottom", size="sm", variant="secondary")
                    
                    text_output = gr.Textbox(
                        label="Edit Book Content", 
                        placeholder="Extracted text will appear here for editing",
                        interactive=True, 
                        lines=15,
                        elem_id="text_editor"
                    )
                    
                    save_btn = gr.Button("Save Edited Text", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown('<div class="step-heading">🎧 Step 3: Generate Audiobook</div>')
                    
                    # TTS Engine selector for audiobook generation
                    # TTS Engine selector for audiobook generation
                    tts_engine_audiobook = gr.Radio(
                        choices=available_engines, # Reuse cached list or call again
                        label="TTS Engine",
                        value=available_engines[0] if available_engines else None,
                        info="Orpheus: Emotion control. Chatterbox: Voice cloning. VibeVoice: High fidelity."
                    )
                    
                    with gr.Row():
                        # Orpheus narrator voice dropdown
                        narrator_voice = gr.Dropdown(
                            choices=voice_choices,
                            label="Narrator Voice (Orpheus)",
                            value="zac",
                            info="Select the voice for narration"
                        )
                        
                        output_format = gr.Dropdown(
                            ["M4B (Chapters & Cover)", "AAC", "M4A", "MP3", "WAV", "OPUS", "FLAC", "PCM"], 
                            label="Output Format",
                            value="M4B (Chapters & Cover)",
                            info="M4B supports chapters and cover art"
                        )
                    
                    # VibeVoice speaker options
                    with gr.Group(visible=False) as vibevoice_audiobook_group:
                        vibevoice_audiobook_selector = gr.Dropdown(
                            choices=get_vibevoice_choices(), # Dynamic choices
                            value=None, # Allow dynamic selection (first item usually)
                            label="VibeVoice Speaker",
                            info="Select a preset speaker or custom voice"
                        )
                        
                        with gr.Row():
                            vibevoice_temperature_audiobook = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                info="Lower = Stable, Higher = Expressive/Random"
                            )
                            vibevoice_top_p_audiobook = gr.Slider(
                                label="Top P",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                info="Lower = Focused, Higher = Diverse"
                            )

                        use_vibevoice_dialogue = gr.Checkbox(
                            label="🎭 Use Separate Dialogue Voice",
                            value=False,
                            info="Use a different voice for quoted dialogue vs narration (VibeVoice)"
                        )
                        
                        vibevoice_dialogue_selector = gr.Dropdown(
                             choices=get_vibevoice_choices(),
                             value=None,
                             label="Dialogue Voice (VibeVoice)",
                             visible=False,
                             info="Select voice for dialogue segments"
                        )

                        def toggle_vibevoice_dialogue(checked):
                            return gr.update(visible=checked)
                        
                        use_vibevoice_dialogue.change(
                            toggle_vibevoice_dialogue,
                            inputs=[use_vibevoice_dialogue],
                            outputs=[vibevoice_dialogue_selector]
                        )

                    # Dialogue voice options (Orpheus only)
                    with gr.Group() as dialogue_voice_group:
                        use_dialogue_voice = gr.Checkbox(
                            label="🎭 Use Separate Dialogue Voice",
                            value=False,
                            info="Use a different voice for quoted dialogue vs narration"
                        )
                        
                        dialogue_voice = gr.Dropdown(
                            choices=voice_choices,
                            label="Dialogue Voice",
                            value="dan",
                            visible=False,
                            info="Voice for dialogue (text in quotes)"
                        )
                    
                    # Chatterbox zero-shot reference audio (hidden by default)
                    with gr.Group(visible=False) as chatterbox_audiobook_group:
                        gr.Markdown("""🎤 **Zero-Shot Voice Cloning with Chatterbox**
                        
                        Upload a ~10 second reference audio sample to clone any voice for your audiobook.
                        """)
                        reference_audio_audiobook = gr.Audio(
                            label="Reference Audio Sample (Required for Chatterbox)",
                            type="filepath",
                            sources=["upload", "microphone"],
                            interactive=True
                        )
                    
                    # Emotion tags checkbox (for Orpheus only)
                    with gr.Group() as emotion_tags_group:
                        add_emotion_tags_checkbox = gr.Checkbox(
                            label="🎭 Add Emotion Tags (Orpheus only)",
                            value=False,
                            interactive=check_llm_availability(),
                            info="Automatically add expressive tags like <laugh>, <sigh>, <gasp> using LLM. (Requires installed LLM)"
                        )
                        
                        with gr.Accordion("What are Emotion Tags?", open=False):
                            gr.Markdown("""
                            **Emotion Tags enhance your audiobook with natural expressions:**

                            * **`<laugh>`** - Laughter | **`<chuckle>`** - Light laugh
                            * **`<sigh>`** - Sighing | **`<gasp>`** - Surprise/shock  
                            * **`<cough>`** - Coughing | **`<groan>`** - Frustration
                            * **`<yawn>`** - Tiredness | **`<sniffle>`** - Sniffling
                            
                            Tags are automatically placed based on text context using an LLM.
                            """)
                    
                    audiobook_postprocess = gr.Checkbox(
                        label="✨ Enhanced Post-processing (Whole Book)", 
                        value=False,
                        info="Run each chapter through audio enhancement (Speech Enhancement + Super-Resolution). Warning: Increases generation time significantly!"
                    )
                    
                    generate_btn = gr.Button("🚀 Generate Audiobook", variant="primary", size="lg")
                    
                    audio_output = gr.Textbox(
                        label="Generation Progress", 
                        placeholder="Generation progress will be shown here",
                        interactive=False,
                        lines=3
                    )
                    
                    # Add a new File component for downloading the audiobook
                    with gr.Group(visible=False) as download_box:
                        gr.Markdown("### 📥 Download Your Audiobook")
                        audiobook_file = gr.File(
                            label="Download Generated Audiobook",
                            interactive=False,
                            type="filepath"
                        )
                    
                    # Hidden state for job ID
                    current_job_id = gr.State(None)
        
        # ==================== Jobs Tab ====================
        with gr.TabItem("📋 Jobs"):
            gr.Markdown("### Audiobook Generation Jobs")
            gr.Markdown("Track your audiobook generation jobs. Jobs are kept for **1 week** so you can reconnect and download your audiobooks.")
            
            with gr.Row():
                refresh_jobs_btn = gr.Button("🔄 Refresh Jobs", variant="secondary")
            
            # Jobs table
            jobs_table = gr.Dataframe(
                headers=["ID", "Book Title", "Engine", "Voice", "Format", "Status", "Progress", "Created", "Expires"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"],
                label="Your Jobs",
                interactive=False,
                type="array",
                wrap=True,
                value=get_jobs_dataframe
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    job_id_input = gr.Textbox(
                        label="Job ID",
                        placeholder="Enter a job ID to view details or download",
                        info="Copy a job ID from the table above"
                    )
                
                with gr.Column(scale=1):
                    view_job_btn = gr.Button("🔍 View Job", variant="primary")
                    download_job_btn = gr.Button("📥 Download", variant="secondary")
                    resume_job_btn = gr.Button("▶️ Resume Job", variant="secondary")
                    delete_job_btn = gr.Button("🗑️ Delete Job", variant="stop")
            
            with gr.Group() as job_details_group:
                gr.Markdown("### Job Details")
                job_details_output = gr.Markdown("Select a job to view details")
                
                with gr.Group(visible=False) as job_download_group:
                    job_download_file = gr.File(
                        label="Download Audiobook",
                        interactive=False,
                        type="filepath"
                    )
            
            # GPU Resource Status Section
            with gr.Accordion("🖥️ GPU Resource Status", open=False):
                gr.Markdown("""
                GPU resources are managed automatically. LLM containers start when needed and stop after being idle.
                This helps free up GPU memory when not generating audiobooks.
                """)
                
                gpu_status_output = gr.Markdown("Click refresh to see GPU status")
                refresh_gpu_btn = gr.Button("🔄 Refresh GPU Status", variant="secondary")
                
                with gr.Row():
                    stop_gpu_btn = gr.Button("🛑 Stop LLM Services Now", variant="stop")
                    start_gpu_btn = gr.Button("🚀 Start LLM Services", variant="primary")
                
                def get_gpu_status():
                    """Get formatted GPU resource status."""
                    status = gpu_manager.get_status()
                    
                    if not status.get("docker_available"):
                        return """
**⚠️ Docker Management Not Available**

Docker socket not mounted. GPU resource management is disabled.
Containers will run continuously (default behavior).

To enable GPU management, ensure docker socket is mounted in docker-compose.yaml.
"""
                    
                    llm_status = "🟢 Running" if status.get("llm_server_running") else "🔴 Stopped"
                    orpheus_status = "🟢 Running" if status.get("orpheus_llama_running") else "🔴 Stopped"
                    vram_used = status.get("estimated_vram_gb", 0)
                    
                    return f"""
**Docker Management:** ✅ Enabled

**Estimated VRAM Usage:** ~{vram_used}GB

**Idle Timeout:** {status.get('idle_timeout', 60)} seconds

### Container Status:
| Service | Status | VRAM |
|---------|--------|------|
| LLM Server (emotion tags) | {llm_status} | ~12GB |
| Orpheus LLM (TTS tokens) | {orpheus_status} | ~5GB |

### GPU Optimization Strategy:
- **Phase 1 (Emotion Tags):** Load LLM server → Process → Unload
- **Phase 2 (TTS):** Load Orpheus → Generate audio → Unload
- **Max VRAM needed:** ~12GB (never both at once!)

*Containers start automatically for each phase and stop immediately after.*
"""
                
                def stop_gpu_services():
                    """Manually stop GPU services."""
                    success, message = gpu_manager.stop_llm_services()
                    return f"{'✅' if success else '❌'} {message}\n\n" + get_gpu_status()
                
                def start_gpu_services():
                    """Manually start GPU services."""
                    success, message = gpu_manager.start_llm_services()
                    return f"{'✅' if success else '❌'} {message}\n\n" + get_gpu_status()
                
                refresh_gpu_btn.click(
                    get_gpu_status,
                    inputs=[
                    book_title, book_input, text_output, 
                    tts_engine_audiobook, narrator_voice, 
                    use_dialogue_voice, dialogue_voice,
                    add_emotion_tags_checkbox,
                    reference_audio_audiobook,
                    vibevoice_audiobook_selector,
                    output_format,
                    audiobook_postprocess,
                    vibevoice_temperature_audiobook,
                    vibevoice_top_p_audiobook,
                    use_vibevoice_dialogue,
                    vibevoice_dialogue_selector
                ],
                    outputs=[gpu_status_output]
                )
                
                stop_gpu_btn.click(
                    stop_gpu_services,
                    inputs=[],
                    outputs=[gpu_status_output]
                )
                
                start_gpu_btn.click(
                    start_gpu_services,
                    inputs=[],
                    outputs=[gpu_status_output]
                )
            
            # Job management functions
            def refresh_jobs():
                """Refresh the jobs table and check for stalled jobs."""
                # Check for jobs that have stalled (no activity for 5 minutes)
                stalled = job_manager.check_for_stalled_jobs(stall_timeout_seconds=300)
                if stalled:
                    print(f"Marked {len(stalled)} jobs as stalled: {stalled}")
                return get_jobs_dataframe()
            
            def view_job_details(job_id):
                """View details of a specific job."""
                if not job_id or not job_id.strip():
                    return "❌ Please enter a job ID", gr.update(visible=False), None
                
                # Check for stalled jobs first
                job_manager.check_for_stalled_jobs(stall_timeout_seconds=300)
                
                job = job_manager.get_job(job_id.strip())
                if not job:
                    return f"❌ Job `{job_id}` not found or has expired", gr.update(visible=False), None
                
                # Format job details
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                    "stalled": "⏸️"
                }.get(job.status, "❓")
                
                details = f"""
**Job ID:** `{job.job_id}`

**Book Title:** {job.book_title}

**TTS Engine:** {job.tts_engine} | **Voice:** {job.voice}

**Output Format:** {job.output_format}

**Status:** {status_emoji} {job.status.replace('_', ' ').title()}

**Progress:** {job.progress}

**Created:** {job.created_at}

**Expires:** {job.expires_at}
"""
                
                if job.error_message:
                    details += f"\n**Error:** {job.error_message}"
                
                # Show resume info for stalled jobs
                if job.status == JobStatus.STALLED.value and job.checkpoint:
                    checkpoint = job.get_checkpoint()
                    if checkpoint:
                        pct = (checkpoint.lines_completed / checkpoint.total_lines * 100) if checkpoint.total_lines > 0 else 0
                        details += f"\n\n**🔄 Resumable:** Yes - {checkpoint.lines_completed}/{checkpoint.total_lines} lines ({pct:.1f}%) completed"
                        details += "\n\n**💡 Tip:** Click the **▶️ Resume Job** button to continue generation from where it stopped."
                
                # Show download if completed
                if job.status == "completed" and job.output_file and os.path.exists(job.output_file):
                    return details, gr.update(visible=True), job.output_file
                else:
                    return details, gr.update(visible=False), None
            
            def delete_job_handler(job_id):
                """Delete a job."""
                if not job_id or not job_id.strip():
                    return "❌ Please enter a job ID", get_jobs_dataframe()
                
                success = job_manager.delete_job(job_id.strip())
                if success:
                    return f"✅ Job `{job_id}` deleted successfully", get_jobs_dataframe()
                else:
                    return f"❌ Job `{job_id}` not found", get_jobs_dataframe()
            
            # Connect job management buttons
            refresh_jobs_btn.click(
                refresh_jobs,
                inputs=[],
                outputs=[jobs_table],
                api_name="get_all_jobs"
            )
            
            # Table row click handler - populate job_id_input when clicking a row
            def on_table_row_select(evt: gr.SelectData, table_data):
                """Handle click on a table row to populate the job ID input."""
                if evt is None or table_data is None:
                    return ""
                try:
                    # evt.index is (row, col), we want the first column (Job ID) of the selected row
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    if row_idx >= 0 and row_idx < len(table_data):
                        job_id = table_data[row_idx][0]  # First column is Job ID
                        return job_id
                except (IndexError, TypeError, KeyError) as e:
                    print(f"Error getting job ID from table selection: {e}")
                return ""
            
            jobs_table.select(
                on_table_row_select,
                inputs=[jobs_table],
                outputs=[job_id_input]
            )
            
            view_job_btn.click(
                view_job_details,
                inputs=[job_id_input],
                outputs=[job_details_output, job_download_group, job_download_file],
                api_name="get_job_details"
            )
            
            delete_job_btn.click(
                delete_job_handler,
                inputs=[job_id_input],
                outputs=[job_details_output, jobs_table],
                api_name="delete_job"
            )
            
            def download_job_file(job_id):
                """Download the output file for a completed job."""
                if not job_id or not job_id.strip():
                    return gr.Warning("Please enter a job ID"), gr.update(visible=False), None
                
                job = job_manager.get_job(job_id.strip())
                if not job:
                    return gr.Warning(f"Job `{job_id}` not found or has expired"), gr.update(visible=False), None
                
                if job.status != "completed":
                    return gr.Warning(f"Job `{job_id}` is not completed yet (status: {job.status})"), gr.update(visible=False), None
                
                if not job.output_file or not os.path.exists(job.output_file):
                    return gr.Warning(f"Output file not found for job `{job_id}`"), gr.update(visible=False), None
                
                return f"📥 **Click the file below to download:** {os.path.basename(job.output_file)}", gr.update(visible=True), job.output_file
            
            download_job_btn.click(
                download_job_file,
                inputs=[job_id_input],
                outputs=[job_details_output, job_download_group, job_download_file],
                api_name="download_job"
            )
            
            # Resume job outputs - need to define components for progress display
            with gr.Group(visible=False) as resume_progress_group:
                resume_progress_output = gr.Textbox(label="Resume Progress", lines=5, interactive=False)
                resume_file_output = gr.File(label="Downloaded Audiobook", visible=False)
            
            # Resume job handler wrapper with auto-retry logic
            async def handle_resume_job(job_id):
                """Handle resume job button click with progress updates and auto-retry on failure."""
                MAX_RETRIES = 3
                results = []
                file_path = None
                
                for attempt in range(MAX_RETRIES):
                    retry_num = job_manager.increment_retry_count(job_id) if attempt > 0 else (job_manager.get_job(job_id).retry_count if job_manager.get_job(job_id) else 0)
                    
                    if attempt > 0:
                        results.append(f"🔄 Retry attempt {attempt + 1}/{MAX_RETRIES} (total retries: {retry_num})...")
                        # Reset endpoints before retry
                        results.append("🔌 Resetting LLM endpoints...")
                        gpu_manager.stop_llm_services()
                        await asyncio.sleep(2)  # Brief pause to ensure clean shutdown
                    
                    success = False
                    error_msg = None
                    
                    async for progress, file, _ in resume_job_wrapper(job_id):
                        if isinstance(progress, str):
                            results.append(progress)
                        if file:
                            file_path = file
                            success = True
                        # Check if this was an error message
                        if progress and "❌" in str(progress):
                            error_msg = str(progress)
                    
                    # If we got a file or job completed successfully, we're done
                    job = job_manager.get_job(job_id)
                    if file_path or (job and job.status == "completed"):
                        # Reset retry count on success
                        job_manager.reset_retry_count(job_id)
                        break
                    
                    # Check if job is now marked as failed (not just stalled)
                    if job and job.status == "failed":
                        results.append(f"❌ Job failed permanently: {job.error_message}")
                        break
                    
                    # If we can retry and haven't exceeded max retries
                    if attempt < MAX_RETRIES - 1 and job_manager.can_auto_retry(job_id):
                        results.append(f"⚠️ Job stalled, will retry automatically...")
                        # Mark as stalled again for the retry
                        job_manager.mark_job_stalled(job_id)
                    else:
                        if attempt == MAX_RETRIES - 1:
                            results.append(f"❌ Max retries ({MAX_RETRIES}) reached. Job remains stalled.")
                        break
                
                # Return final state
                progress_text = "\n".join(results[-15:]) if results else "No progress"  # Last 15 lines
                return progress_text, file_path, get_jobs_dataframe()
            
            resume_job_btn.click(
                handle_resume_job,
                inputs=[job_id_input],
                outputs=[job_details_output, job_download_file, jobs_table],
                api_name="resume_job"
            )

        # ==================== Model Manager Tab ====================
        with gr.TabItem("🧠 Model Manager"):
            gr.Markdown("### AI Model Manager")
            gr.Markdown("Manage TTS, LLM, and Transcription models. Download models here to use them.")
            
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
            
            def get_model_list_for_ui():
                """Get list of models for Gradio Dataframe."""
                statuses = model_manager.list_models()
                data = []
                for s in statuses:
                    status_str = "✅ Installed" if s.installed else "❌ Not Installed"
                    if s.installed and s.downloaded_size_gb > 0:
                        status_str += f" ({s.downloaded_size_gb:.2f} GB)"
                    
                    data.append([
                        s.id,
                        s.definition.name,
                        s.definition.type.value.upper(),
                        status_str,
                        f"{s.definition.size_estimate_gb} GB"
                    ])
                return data

            models_table = gr.Dataframe(
                headers=["ID", "Name", "Type", "Status", "Size Estimate"],
                datatype=["str", "str", "str", "str", "str"],
                label="Supported Models",
                interactive=False,
                type="array",
                wrap=True,
                value=get_model_list_for_ui
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    model_details_output = gr.Markdown("Select a model from the table to manage it.")
                with gr.Column(scale=1):
                    # Contextual action button (Download or Delete)
                    model_action_btn = gr.Button("Select Model", variant="secondary", interactive=False)
            
            # Hidden state to store selected model ID
            selected_model_id = gr.State(None)
            
            def on_model_select(evt: gr.SelectData, table_data):
                if evt is None: return "Select a model", gr.update(value="Select Model", interactive=False), None
                try:
                    row_idx = evt.index[0]
                    model_id = table_data[row_idx][0]
                    
                    # Check status
                    path = model_manager.get_model_path(model_id)
                    installed = path is not None
                    
                    details = f"**Selected:** `{model_id}`\n\n**Status:** {'✅ Installed' if installed else '❌ Not Installed'}"
                    if installed:
                        return details, gr.update(value="🗑️ Delete Model", variant="stop", interactive=True), model_id
                    else:
                        return details, gr.update(value="⬇️ Download Model", variant="primary", interactive=True), model_id
                except Exception as e:
                    print(f"Error selecting model: {e}")
                    return f"Error: {e}", gr.update(value="Error", interactive=False), None
                
            def handle_model_action(model_id, current_label):
                if not model_id: return "⚠️ Select a model first", gr.update()
                
                if "Download" in current_label:
                    import threading
                    def _bg():
                        try:
                            print(f"Starting download for {model_id}...")
                            model_manager.download_model(model_id)
                            print(f"Download complete for {model_id}")
                        except Exception as e:
                            print(f"Download error for {model_id}: {e}")
                    threading.Thread(target=_bg).start()
                    return f"⏳ Started background download for {model_id}.", gr.update(value="⏳ Downloading...", interactive=False)
                
                elif "Delete" in current_label:
                    if model_manager.delete_model(model_id):
                        return f"✅ Deleted {model_id}", gr.update(value="⬇️ Download Model", variant="primary", interactive=True)
                    return f"❌ Failed to delete {model_id}", gr.update()
                
                return "Unknown action", gr.update()

            # Wiring
            models_table.value = get_model_list_for_ui
            refresh_models_btn.click(fn=get_model_list_for_ui, inputs=[], outputs=[models_table])
            
            models_table.select(
                fn=on_model_select, 
                inputs=[models_table], 
                outputs=[model_details_output, model_action_btn, selected_model_id]
            )
            
            model_action_btn.click(
                fn=handle_model_action,
                inputs=[selected_model_id, model_action_btn], 
                outputs=[model_details_output, model_action_btn]
            ).then(
                fn=get_model_list_for_ui,
                outputs=[models_table]
            )

        # ==================== Voice Library Tab ====================
        with gr.TabItem("🎤 Voice Library"):
            gr.Markdown("### Custom Voice Library")
            gr.Markdown("Upload reference audio files for zero-shot voice cloning (Use with **Chatterbox** or **VibeVoice**).")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Add New Voice")
                    new_voice_name = gr.Textbox(label="Voice Name", placeholder="e.g. My Custom Voice")
                    new_voice_file = gr.Audio(label="Reference Audio", type="filepath", sources=["upload", "microphone"])
                    
                    with gr.Group():
                        enable_preprocessing = gr.Checkbox(
                            label="✨ Enhanced Preprocessing", 
                            value=True,
                            info="Run through Demucs (vocal isolation) + ClearerVoice (enhancement) + Super-Resolution (48kHz)"
                        )
                        enable_smart_selection = gr.Checkbox(
                            label="🧠 Smart Selection (Whisper)",
                            value=False,
                            info="Use Whisper AI to select best ~90s of speech (good for long files)"
                        )
                        target_duration_slider = gr.Slider(
                            minimum=10, 
                            maximum=300, 
                            value=90, 
                            step=10, 
                            label="Target Duration (seconds)",
                            info="Target length for smart selection",
                            visible=False
                        )
                        
                        # Show slider only when smart selection is enabled
                        def toggle_duration_slider(enabled):
                            return gr.update(visible=enabled)
                            
                        enable_smart_selection.change(
                            fn=toggle_duration_slider,
                            inputs=[enable_smart_selection],
                            outputs=[target_duration_slider]
                        )
                    
                    add_voice_btn = gr.Button("💾 Save Voice", variant="primary")
                    add_voice_status = gr.Markdown("")
                
                with gr.Column(scale=2):
                    gr.Markdown("#### Saved Voices")
                    refresh_voices_btn = gr.Button("🔄 Refresh List", variant="secondary")
                    
                    def get_voices_dataframe():
                        voices = voice_manager.list_voices()
                        return [[v.name, v.id] for v in voices]
                        
                    voices_table = gr.Dataframe(
                        headers=["Name", "Filename"],
                        datatype=["str", "str"],
                        label="Your Voices",
                        interactive=False,
                        type="array",
                        value=get_voices_dataframe
                    )
                    
                    with gr.Row():
                         delete_voice_input = gr.Textbox(label="Filename to Delete", placeholder="Select from table or type filename")
                         delete_voice_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                    
                    voice_preview_player = gr.Audio(label="🎧 Selected Voice Preview", interactive=False)
                    
                    delete_status = gr.Markdown("")

            async def upload_voice_handler(name, file, preprocess, smart_select, duration):
                if not name or not file:
                    yield "Please provide name and file.", gr.update(value=get_voices_dataframe()), gr.update(), gr.update(), gr.update(), gr.update()
                    return
                
                info = f"Processing with: {'Enhanced Preprocessing' if preprocess else ''} {'+ Smart Selection' if smart_select else ''}"
                yield f"⏳ Uploading and processing voice... {info}", gr.update(value=get_voices_dataframe()), gr.update(), gr.update()
                
                # Double check input
                if not name or not file:
                     yield "❌ Invalid input.", gr.update(value=get_voices_dataframe()), gr.update(), gr.update(), gr.update(), gr.update()
                     return

                result = await voice_manager.add_voice(
                    name, 
                    file, 
                    preprocess=preprocess, 
                    smart_selection=smart_select,
                    target_duration=duration
                )
                
                if result:
                    voice_info, status = result
                    choices = get_vibevoice_choices()
                    voice_choices = get_voice_choices() 
                    # Refresh all dropdowns that depend on voices
                    yield f"✅ Success: {status}", gr.update(value=get_voices_dataframe()), gr.update(choices=choices), gr.update(choices=voice_choices), gr.update(choices=choices), gr.update(choices=choices)
                else:
                    yield "❌ Upload failed. Check logs.", gr.update(value=get_voices_dataframe()), gr.update(), gr.update(), gr.update(), gr.update()

            add_voice_btn.click(
                fn=upload_voice_handler,
                inputs=[new_voice_name, new_voice_file, enable_preprocessing, enable_smart_selection, target_duration_slider],
                outputs=[add_voice_status, voices_table, vibevoice_selector, voice_selector, vibevoice_audiobook_selector, vibevoice_dialogue_selector]
            )
            
            def delete_voice_handler(filename):
                if voice_manager.delete_voice(filename):
                    status = f"✅ Deleted {filename}"
                    choices = get_vibevoice_choices()
                    return status, get_voices_dataframe(), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
                return f"❌ Failed to delete {filename}", gr.update(), gr.update(), gr.update(), gr.update()
            
            # Select from table to fill delete input AND preview player
            def on_voice_select(evt: gr.SelectData, data):
                row = evt.index[0]
                filename = data[row][1] # Filename
                
                # Find path for preview
                voice_path = None
                for v in voice_manager.list_voices():
                    if v.id == filename:
                        voice_path = v.path
                        break
                return filename, voice_path
            
            
            def refresh_library_handler():
                choices = get_vibevoice_choices()
                return get_voices_dataframe(), gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
            
            refresh_voices_btn.click(
                refresh_library_handler, 
                outputs=[voices_table, vibevoice_selector, vibevoice_audiobook_selector, vibevoice_dialogue_selector]
            )
            
            voices_table.select(
                on_voice_select, 
                inputs=[voices_table], 
                outputs=[delete_voice_input, voice_preview_player]
            )
            
            delete_voice_btn.click(
                delete_voice_handler,
                inputs=[delete_voice_input],
                outputs=[delete_status, voices_table, vibevoice_selector, vibevoice_audiobook_selector, vibevoice_dialogue_selector]
            )
            
            # Update all voice dropdowns when voices change (Global Refresh)
            # We can't easily trigger updates across tabs without global events or refresh buttons in those tabs.
            # But the dropdowns populate on app load. 
            # We can enable the refresh button in other tabs to reload choices.
            
            def update_voice_dropdowns():
                return gr.update(choices=get_voice_choices())

            # Link refresh to dropdown updates in other tabs?
            # Or just add a "Refresh Voices" button in Sampling tab.



    # Connections with proper handling of Gradio notifications
    
    # TTS engine change handler for audiobook tab - toggle visibility
    def update_audiobook_tts_visibility(tts_engine):
        if tts_engine == "Orpheus":
            # Show Orpheus controls, hide others
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update()
        elif tts_engine == "VibeVoice":
            # Show VibeVoice controls AND refresh choices
            new_choices = get_vibevoice_choices()
            # If no choices, default might be None or first item?
            # We keep current value if valid, but here we just refresh choices.
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(choices=new_choices), gr.update(choices=new_choices)
        else:  # Chatterbox
            # Show Chatterbox controls
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update()
    
    tts_engine_audiobook.change(
        update_audiobook_tts_visibility,
        inputs=[tts_engine_audiobook],
        outputs=[narrator_voice, chatterbox_audiobook_group, emotion_tags_group, dialogue_voice_group, vibevoice_audiobook_group, vibevoice_audiobook_selector, vibevoice_dialogue_selector]
    )
    
    # Dialogue voice checkbox handler - show/hide dialogue voice dropdown
    def update_dialogue_voice_visibility(use_dialogue):
        return gr.update(visible=use_dialogue)
    
    use_dialogue_voice.change(
        update_dialogue_voice_visibility,
        inputs=[use_dialogue_voice],
        outputs=[dialogue_voice]
    )
    
    # Auto-fill book title from uploaded filename
    book_input.change(
        extract_title_from_filename,
        inputs=[book_input],
        outputs=[book_title]
    )
    
    validate_btn.click(
        validate_book_upload, 
        inputs=[book_input, book_title], 
        outputs=[book_title],
        api_name="validate_book"
    )
    
    convert_btn.click(
        text_extraction_wrapper, 
        inputs=[book_input], 
        outputs=[text_output],
        queue=True,
        api_name="extract_text"
    )
    
    save_btn.click(
        save_book_wrapper, 
        inputs=[text_output], 
        outputs=[],
        queue=True,
        api_name="save_text"
    )

    # Dummy load event to update engine visibility on app start
    gradio_app.load(
        fn=update_tts_engine_visibility,
        inputs=[tts_engine_sampling],
        outputs=[orpheus_voice_group, chatterbox_voice_group, vibevoice_voice_group]
    )
    
    # Update the generate_audiobook_wrapper to output progress text, file path, and job ID
    # Now includes automatic emotion tag processing if checkbox is enabled
    generate_btn.click(
        generate_audiobook_wrapper, 
        inputs=[tts_engine_audiobook, narrator_voice, output_format, book_input, add_emotion_tags_checkbox, book_title, use_dialogue_voice, dialogue_voice, reference_audio_audiobook, vibevoice_audiobook_selector, audiobook_postprocess, vibevoice_temperature_audiobook, vibevoice_top_p_audiobook, use_vibevoice_dialogue, vibevoice_dialogue_selector], 
        outputs=[audio_output, audiobook_file, current_job_id],
        queue=True,
        api_name="generate_audiobook"
    ).then(
        # Make the download box visible after generation completes successfully
        lambda x: gr.update(visible=True) if x is not None else gr.update(visible=False),
        inputs=[audiobook_file],
        outputs=[download_box]
    ).then(
        # Refresh the jobs table after generation
        lambda: get_jobs_dataframe(),
        inputs=[],
        outputs=[jobs_table]
    )
    
    # Navigation button functionality for textbox scrolling
    top_btn.click(
        None,
        inputs=[],
        outputs=[],
        js="""
        function() {
            const textbox = document.querySelector('#text_editor textarea');
            if (textbox) {
                textbox.scrollTop = 0;
            }
        }
        """
    )
    
    bottom_btn.click(
        None,
        inputs=[],
        outputs=[],
        js="""
        function() {
            const textbox = document.querySelector('#text_editor textarea');
            if (textbox) {
                textbox.scrollTop = textbox.scrollHeight;
            }
        }
        """
    )

app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)