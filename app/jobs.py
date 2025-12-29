"""
Background job runners for Audiobook Creator.

Contains async functions that run audiobook generation jobs independently of Gradio.
Extracted from app.py for modularity.
"""

import os
import shutil
import traceback
from audiobook.core.emotion_tags import process_emotion_tags
from audiobook.tts.generator import process_audiobook_generation
from audiobook.utils.job_manager import job_manager
from audiobook.utils.gpu_resource_manager import gpu_manager


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
    vibevoice_dialogue_voice: str = None
):
    """
    Background async function that runs the audiobook generation.
    This runs independently of the Gradio connection.
    Progress is saved to job_manager so users can check status even after closing browser.
    """
    need_orpheus_tts = tts_engine == "Orpheus"
    
    try:
        # PHASE 1: Emotion Tags (if enabled) - Uses LLM server (~12GB VRAM)
        if add_emotion_tags:
            job_manager.update_job_progress(job_id, "🚀 Starting LLM server for emotion tags...")
            
            success, gpu_message = gpu_manager.acquire_llm()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start LLM server: {gpu_message}")
                return
            
            try:
                job_manager.update_job_progress(job_id, "🎭 Processing emotion tags...")
                
                async for progress in process_emotion_tags(False):
                    job_manager.update_job_progress(job_id, f"🎭 Emotion tags: {str(progress)[:100]}")
                
                if os.path.exists("tag_added_lines_chunks.txt"):
                    job_emotion_tags_path = job_manager.get_job_emotion_tags_path(job_id)
                    shutil.copy2("tag_added_lines_chunks.txt", job_emotion_tags_path)
                
                job_manager.update_job_progress(job_id, "✅ Emotion tags added successfully!")
            except Exception as e:
                job_manager.fail_job(job_id, f"Emotion tags failed: {str(e)}")
                traceback.print_exc()
                return
            finally:
                print(f"🔄 Job {job_id}: Releasing LLM server to free VRAM for TTS...")
                gpu_manager.release_llm(immediate=True)
        
        # PHASE 2: TTS Generation - Uses Orpheus LLM (~5GB VRAM) for Orpheus engine
        if need_orpheus_tts:
            job_manager.update_job_progress(job_id, "🚀 Starting Orpheus LLM for TTS...")
            
            success, gpu_message = gpu_manager.acquire_orpheus()
            if not success:
                job_manager.fail_job(job_id, f"Failed to start Orpheus LLM: {gpu_message}")
                return
        
        try:
            job_manager.update_job_progress(job_id, "Starting audiobook generation...")
            
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
                if output:
                    job_manager.update_job_progress(job_id, str(output)[:200])
            
            generate_m4b_audiobook_file = output_format == "M4B (Chapters & Cover)"
            file_extension = "m4b" if generate_m4b_audiobook_file else output_format.lower()
            
            audiobook_path = os.path.join("generated_audiobooks", f"audiobook.{file_extension}")

            final_path = os.path.join("generated_audiobooks", f"{book_title}.{file_extension}")
            if os.path.exists(audiobook_path):
                os.rename(audiobook_path, final_path)
                audiobook_path = final_path
            
            job_manager.complete_job(job_id, audiobook_path)
            print(f"✅ Job {job_id}: Audiobook generated successfully!")
            
        finally:
            if need_orpheus_tts:
                print(f"🔓 Job {job_id}: Releasing Orpheus LLM")
                gpu_manager.release_orpheus()
                
    except Exception as e:
        print(f"❌ Job {job_id}: Error - {e}")
        traceback.print_exc()
        job_manager.fail_job(job_id, str(e))


async def resume_job_background(job_id: str, job, checkpoint):
    """
    Background async function that resumes a stalled audiobook generation job.
    This runs independently of the Gradio connection.
    """
    need_orpheus_tts = job.tts_engine == "Orpheus"
    add_emotion_tags = checkpoint.add_emotion_tags
    
    job_converted_book_path = job_manager.get_job_converted_book_path(job_id)
    job_emotion_tags_path = job_manager.get_job_emotion_tags_path(job_id)
    
    try:
        # Check if emotion tags need to be re-processed
        emotion_tags_exist = os.path.exists(job_emotion_tags_path) or os.path.exists("tag_added_lines_chunks.txt")
        if add_emotion_tags and not emotion_tags_exist:
            job_manager.update_job_progress(job_id, "🎭 Re-processing emotion tags...")
            
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
                
                import subprocess
                result = subprocess.run(
                    ["ebook-convert", book_path, job_converted_book_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    job_manager.fail_job(job_id, f"Failed to extract text: {result.stderr}")
                    return
                
                shutil.copy2(job_converted_book_path, "converted_book.txt")
            
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
            
            generate_m4b_audiobook_file = job.output_format == "M4B (Chapters & Cover)"
            file_extension = "m4b" if generate_m4b_audiobook_file else job.output_format.lower()
            
            audiobook_path = os.path.join("generated_audiobooks", f"audiobook.{file_extension}")
            
            final_path = os.path.join("generated_audiobooks", f"{job.book_title}.{file_extension}")
            if os.path.exists(audiobook_path):
                os.rename(audiobook_path, final_path)
                audiobook_path = final_path
            
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
