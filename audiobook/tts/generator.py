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

import shutil
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Any, Tuple, Set
import json
import os
import asyncio
import re
import tempfile
from word2number import w2n
import time
import sys
from pydub import AudioSegment
from audiobook.utils.shell_commands import check_if_ffmpeg_is_installed, check_if_calibre_is_installed
from audiobook.utils.file_utils import read_json, empty_directory
from audiobook.tts.audio_utils import merge_chapters_to_m4b, convert_audio_file_formats, add_silence_to_audio_file_by_reencoding_using_ffmpeg, merge_chapters_to_standard_audio_file, add_silence_to_audio_file_by_appending_pre_generated_silence, assemble_chapter_with_ffmpeg, add_silence_to_chapter_with_ffmpeg, get_ebook_metadata_with_cover, validate_file_path, concatenate_audio_files_with_ffmpeg
from audiobook.utils.api_health import check_if_audio_generator_api_is_up
from audiobook.tts.voice_mapping import get_narrator_and_dialogue_voices
from audiobook.utils.text_preprocessing import preprocess_text_for_tts
from audiobook.utils.llm_utils import generate_audio_with_engine
from audiobook.utils.job_manager import job_manager, JobCheckpoint, JobProgress
from audiobook.utils.logging_config import get_logger
from audiobook.config import settings

logger = get_logger(__name__)

TTS_BASE_URL = settings.tts_base_url
TTS_API_KEY = settings.tts_api_key
TTS_MODEL = "orpheus"  # Always use Orpheus TTS
TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE = settings.tts_max_parallel_requests_batch_size

os.makedirs("audio_samples", exist_ok=True)
os.makedirs("generated_audiobooks", exist_ok=True)

async_openai_client = AsyncOpenAI(
    base_url=TTS_BASE_URL, api_key=TTS_API_KEY
)

def sanitize_filename(text: str) -> str:
    # Remove or replace problematic characters
    text = text.replace("'", '').replace('"', '').replace('/', ' ').replace('.', ' ')
    text = text.replace(':', '').replace('?', '').replace('\\', '').replace('|', '')
    text = text.replace('*', '').replace('<', '').replace('>', '').replace('&', 'and')
    
    # cleanup file name based on pattern in run_shell_command_secure
    # ^[a-zA-Z0-9\-_./]+$
    regex = r"[^a-zA-Z0-9\-_./\s]"
    text = re.sub(regex, ' ', text, 0, re.MULTILINE) 
    
    # Normalize whitespace and trim
    text = ' '.join(text.split())
    
    return text

def is_only_punctuation(text):
    """
    Check if a line contains only punctuation marks without any actual words.
    This helps avoid TTS errors when encountering lines with just punctuation.
    
    Args:
        text (str): The text line to check
        
    Returns:
        bool: True if the line contains only punctuation, False otherwise
    """
    # Remove all whitespace
    cleaned_text = text.strip()
    
    # If empty after stripping, it's not useful for TTS
    if not cleaned_text:
        return True
    
    # Import string for standard punctuation
    import string
    
    # Extended punctuation set including common Unicode punctuation in books
    extended_punctuation = string.punctuation + '—–""''…‚„‹›«»‰‱'
    
    # Remove all punctuation marks (both ASCII and extended Unicode)
    text_without_punct = ''.join(char for char in cleaned_text if char not in extended_punctuation)
    
    # If nothing remains after removing punctuation, it's only punctuation
    return len(text_without_punct.strip()) == 0

def split_and_annotate_text(text):
    """Splits text into dialogue and narration while annotating each segment."""
    parts = re.split(r'("[^"]+")', text)  # Keep dialogues in the split result
    annotated_parts = []

    for part in parts:
        if part:  # Ignore empty strings
            annotated_parts.append({
                "text": part,
                "type": "dialogue" if part.startswith('"') and part.endswith('"') else "narration"
            })

    return annotated_parts

def check_if_chapter_heading(text):
    """
    Checks if a given text line represents a chapter heading.

    A chapter heading is considered a string that starts with either "Chapter",
    "Part", or "PART" (case-insensitive) followed by a number (either a digit
    or a word that can be converted to an integer).

    :param text: The text to check
    :return: True if the text is a chapter heading, False otherwise
    """
    pattern = r'^(Chapter|Part|PART)\s+([\w-]+|\d+)'
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.match(text)

    if match:
        label, number = match.groups()
        try:
            # Try converting the number (either digit or word) to an integer
            w2n.word_to_num(number) if not number.isdigit() else int(number)
            return True
        except ValueError:
            return False  # Invalid number format
    return False  # No match
    
def find_voice_for_gender_score(character: str, character_gender_map, engine_name: str, narrator_gender: str):
    """
    Finds the appropriate voice for a character based on their gender score using the new voice mapping system.

    This function takes in the name of a character, a dictionary mapping character names to their gender scores,
    the TTS engine name, and the narrator gender preference. It returns the voice identifier that matches 
    the character's gender score within the appropriate score map (male_score_map or female_score_map).

    Args:
        character (str): The name of the character for whom the voice is being determined.
        character_gender_map (dict): A dictionary mapping character names to their gender scores.
        engine_name (str): The TTS engine name (always "orpheus").
        narrator_gender (str): User's narrator gender preference ("male" or "female").

    Returns:
        str: The voice identifier that matches the character's gender score.
    """

    # Handle narrator character specially
    if character.lower() == "narrator":
        return get_narrator_voice_for_character(engine_name, narrator_gender)

    # Get the character's gender score
    if "scores" in character_gender_map and character.lower() in character_gender_map["scores"]:
        character_info = character_gender_map["scores"][character.lower()]
        character_gender_score = character_info["gender_score"]
        
        return get_voice_for_character_score(engine_name, narrator_gender, character_gender_score)
    else:
        # Fallback for unknown characters - use score 5 (neutral)
        return get_voice_for_character_score(engine_name, narrator_gender, 5)

def validate_book_for_m4b_generation(book_path):
    """
    Validates that the book file is suitable for M4B audiobook generation.
    
    This function performs early validation to catch issues before audio generation:
    - Checks if the book file path is safe and accessible
    - Verifies that ebook-meta command is available
    - Tests metadata extraction from the book
    - Ensures cover image can be extracted
    
    Args:
        book_path (str): Path to the book file
        
    Returns:
        tuple: (is_valid, error_message, metadata)
            - is_valid (bool): True if validation passed
            - error_message (str): Error description if validation failed, None if passed
            - metadata (dict): Extracted metadata if successful, None if failed
    """
    try:
        # Validate file path safety and existence
        if not validate_file_path(book_path):
            return False, f"Invalid or inaccessible book file: {book_path}. Please check the file path and permissions.", None
        
        # Test metadata extraction (this also validates ebook-meta availability)
        metadata = get_ebook_metadata_with_cover(book_path)
        
        # Check if we got meaningful metadata
        if not metadata or len(metadata) == 0:
            return False, f"No metadata could be extracted from the book file: {book_path}. Please ensure it's a valid ebook format.", None
            
        # Check if cover extraction worked (cover.jpg should exist after get_ebook_metadata_with_cover)
        if not validate_file_path("cover.jpg"):
            return False, f"Could not extract cover image from the book file: {book_path}. The book may not contain a cover image.", None
            
        return True, None, metadata
        
    except ValueError as e:
        return False, f"Book file validation error: {str(e)}", None
    except RuntimeError as e:
        return False, f"Ebook processing error: {str(e)}. Please ensure Calibre is properly installed and the book file is not corrupted.", None
    except Exception as e:
        return False, f"Unexpected error during book validation: {str(e)}", None

async def generate_audio_with_single_voice(
    output_format: str, 
    narrator_voice: str, 
    generate_m4b_audiobook_file: bool = False, 
    book_path: str = "", 
    add_emotion_tags: bool = False, 
    job_id: Optional[str] = None, 
    resume_checkpoint: Optional[Dict] = None, 
    dialogue_voice: Optional[str] = None, 
    use_postprocessing: bool = False, 
    tts_engine: str = "Orpheus", 
    vibevoice_voice: Optional[str] = None, 
    vibevoice_temperature: float = 0.7, 
    vibevoice_top_p: float = 0.95, 
    use_vibevoice_dialogue: bool = False, 
    vibevoice_dialogue_voice: Optional[str] = None, 
    reference_audio_path: Optional[str] = None
):
    # Read the text from the file
    """
    Generate an audiobook using a single voice for narration and optionally a separate voice for dialogues.

    Args:
        ... (existing args)
        use_postprocessing (bool, optional): Whether to run audio enhancement on each chapter. Defaults to False.
    """
    # Import job manager for checkpoint saving and job-specific paths
    from audiobook.utils.job_manager import job_manager, JobCheckpoint
    
    # Determine if we're resuming
    is_resuming = resume_checkpoint is not None
    
    # Setup job-specific directories
    if job_id:
        temp_audio_dir = job_manager.get_job_dir(job_id)
        temp_line_audio_dir = job_manager.get_job_line_segments_dir(job_id)
        converted_book_path = job_manager.get_job_converted_book_path(job_id)
        emotion_tags_path = job_manager.get_job_emotion_tags_path(job_id)
    else:
        # Fallback for non-job runs (shouldn't happen in normal use)
        temp_audio_dir = "temp_audio"
        temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")
        converted_book_path = "converted_book.txt"
        emotion_tags_path = "tag_added_lines_chunks.txt"
    
    # Early validation for M4B generation (skip if resuming - already validated)
    if generate_m4b_audiobook_file and not is_resuming:
        yield "Validating book file for M4B audiobook generation..."
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_path)
        
        if not is_valid:
            raise ValueError(f"❌ Book validation failed: {error_message}")
            
        yield f"✅ Book validation successful! Title: {metadata.get('Title', 'Unknown')}, Author: {metadata.get('Author(s)', 'Unknown')}"
        
        # Copy book file to persistent location for resume capability
        if book_path and os.path.exists(book_path):
            book_filename = os.path.basename(book_path)
            persistent_book_path = os.path.join("generated_audiobooks", f"_temp_{book_filename}")
            shutil.copy2(book_path, persistent_book_path)
            book_path = persistent_book_path
            yield f"📁 Book file copied to persistent storage for resume capability"
    elif is_resuming and generate_m4b_audiobook_file:
        # When resuming, check if the book path exists, if not try the persistent copy
        if not os.path.exists(book_path):
            # Try to find the persistent copy
            book_filename = os.path.basename(book_path)
            persistent_book_path = os.path.join("generated_audiobooks", f"_temp_{book_filename}")
            if os.path.exists(persistent_book_path):
                book_path = persistent_book_path
                yield f"📁 Using persistent book file copy for M4B generation"
            else:
                yield f"⚠️ Original book file not found - M4B will be created without cover/metadata"

    # Check if emotion tags should be used and if they have been pre-applied
    # First check job-specific path, then fallback to root path for backwards compatibility
    if add_emotion_tags:
        if os.path.exists(emotion_tags_path):
            with open(emotion_tags_path, "r", encoding='utf-8') as f:
                text = f.read()
            yield f"Using pre-processed text with emotion tags from job directory"
        elif os.path.exists("tag_added_lines_chunks.txt"):
            # Backwards compatibility: copy to job directory
            with open("tag_added_lines_chunks.txt", "r", encoding='utf-8') as f:
                text = f.read()
            # Save to job-specific path
            with open(emotion_tags_path, "w", encoding='utf-8') as f:
                f.write(text)
            yield "Using pre-processed text with emotion tags (migrated to job directory)"
        else:
            # No emotion tags file exists, fall through to regular text
            add_emotion_tags = False
    
    if not add_emotion_tags:
        # Try job-specific converted book first, then root, then original
        if os.path.exists(converted_book_path):
            with open(converted_book_path, "r", encoding='utf-8') as f:
                text = f.read()
            yield f"Loaded converted book from job directory"
        elif os.path.exists("converted_book.txt"):
            with open("converted_book.txt", "r", encoding='utf-8') as f:
                text = f.read()
            # Copy to job directory for persistence
            with open(converted_book_path, "w", encoding='utf-8') as f:
                f.write(text)
            yield "Loaded converted book (copied to job directory)"
        else:
            raise FileNotFoundError(f"No converted book file found at {converted_book_path} or converted_book.txt")
        
        # Apply text preprocessing for Orpheus TTS to prevent repetition issues
        if TTS_MODEL.lower() == "orpheus":
            text = preprocess_text_for_tts(text)
            yield "Applied text preprocessing for Orpheus TTS"
    
    lines = text.split("\n")
    
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Voice configuration
    # Voice configuration
    engine_lower = tts_engine.lower()
    
    # Determine effective voices based on engine
    effective_narrator = narrator_voice
    effective_dialogue = dialogue_voice
    
    if engine_lower == "vibevoice":
        effective_narrator = vibevoice_voice if vibevoice_voice else "speaker_0"
        if use_vibevoice_dialogue and vibevoice_dialogue_voice:
             effective_dialogue = vibevoice_dialogue_voice
        else:
             effective_dialogue = None # Disable dialogue splitting if not enabled for VibeVoice

    use_separate_dialogue = effective_dialogue is not None and effective_dialogue != effective_narrator
    
    if use_separate_dialogue:
        yield f"Using narrator voice: {effective_narrator}, dialogue voice: {effective_dialogue} ({tts_engine})"
    else:
        yield f"Using voice: {effective_narrator} ({tts_engine})"

    # Ensure job directories exist (already created by JobManager but ensure they exist)
    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)

    # Only clear directories if NOT resuming
    if not resume_checkpoint:
        empty_directory(temp_audio_dir)
        # Recreate after clearing
        os.makedirs(temp_audio_dir, exist_ok=True)
        os.makedirs(temp_line_audio_dir, exist_ok=True)
    
    # Batch processing parameters
    # For VibeVoice (7B model), we limit concurrency to 1 to avoid OOM
    if engine_lower == "vibevoice":
        effective_batch_size = 1
    else:
        effective_batch_size = TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE
        
    semaphore = asyncio.Semaphore(effective_batch_size)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.wav"
    chapter_files = []
    
    # Check if M4A chapter files already exist (resume at merge step)
    existing_m4a_files = []
    if resume_checkpoint and os.path.exists(temp_audio_dir):
        for f in sorted(os.listdir(temp_audio_dir)):
            if f.endswith('.m4a'):
                existing_m4a_files.append(f)
    
    if existing_m4a_files:
        yield f"🔄 Found {len(existing_m4a_files)} existing M4A chapter files - skipping to merge step"
        
        # Skip directly to final merge
        if generate_m4b_audiobook_file:
            yield "Creating M4B audiobook file..."
            merge_chapters_to_m4b(book_path, existing_m4a_files, temp_audio_dir)
            yield "M4B audiobook created successfully"
        else:
            yield "Creating final audiobook..."
            merge_chapters_to_standard_audio_file(existing_m4a_files, temp_audio_dir)
            convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
            yield f"Audiobook in {output_format} format created successfully"
        return
    
    # First pass: Generate audio for each line independently
    total_size = len(lines)

    progress_counter = 0
    
    # Check for existing line audio files (for resume support)
    # Primary source of truth is now the database, not filesystem
    existing_lines = set()
    if job_id:
        # Get completed lines from database (source of truth)
        existing_lines = job_manager.get_completed_lines(job_id)
    
    # Fallback: also check filesystem for backward compatibility with old jobs
    if not existing_lines and (resume_checkpoint or os.path.exists(temp_line_audio_dir)):
        for f in os.listdir(temp_line_audio_dir) if os.path.exists(temp_line_audio_dir) else []:
            if f.startswith("line_") and f.endswith(".wav"):
                try:
                    line_idx = int(f.replace("line_", "").replace(".wav", ""))
                    existing_lines.add(line_idx)
                except ValueError:
                    pass
    
    if existing_lines:
        yield f"🔄 Resuming: Found {len(existing_lines)} previously generated lines"
        progress_counter = len(existing_lines)
    
    # For tracking progress with tqdm in an async context
    progress_bar = tqdm(total=total_size, unit="line", desc="Audio Generation Progress", initial=progress_counter)
    
    # Maps chapters to their line indices
    chapter_line_map = {}
    
    # Checkpoint save interval (save every N lines)
    CHECKPOINT_INTERVAL = 50
    last_checkpoint_count = progress_counter
    
    # Track failed lines for retry
    failed_lines = {}  # {line_index: (line_text, error_count)}
    MAX_RETRIES_PER_LINE = 5
    RETRY_DELAY_BASE = 2  # seconds
    
    async def generate_line_audio(line_index, line, text_to_speak, voice, line_audio_path):
        """Generate audio for a single piece of text with retry logic."""
        last_error = None
        for attempt in range(MAX_RETRIES_PER_LINE):
            try:
                async with semaphore:
                    audio_buffer = await generate_audio_with_engine(
                        text_to_speak, 
                        voice,
                        engine=engine_lower,
                        temperature=vibevoice_temperature,
                        top_p=vibevoice_top_p,
                        reference_audio=reference_audio_path
                    )
                return audio_buffer
            except Exception as e:
                last_error = e
                delay = RETRY_DELAY_BASE * (attempt + 1)
                print(f"⚠️ Line {line_index} attempt {attempt + 1}/{MAX_RETRIES_PER_LINE} failed: {str(e)[:100]}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        raise last_error if last_error else Exception("Unknown error generating audio")

    async def process_single_line(line_index, line):
        nonlocal progress_counter, last_checkpoint_count

        if not line or is_only_punctuation(line):
            progress_bar.update(1)
            progress_counter += 1
            return {"index": line_index, "is_chapter_heading": False, "line": line, "skipped": True}
        
        # Check if this line was already generated (resume support)
        line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.wav")
        if line_index in existing_lines and os.path.exists(line_audio_path):
            # Verify file is valid (not empty/corrupted)
            try:
                file_size = os.path.getsize(line_audio_path)
                if file_size > 1000:  # Valid WAV should be > 1KB
                    progress_bar.update(1)
                    progress_counter += 1
                    return {
                        "index": line_index,
                        "is_chapter_heading": check_if_chapter_heading(line),
                        "line": line,
                    }
                else:
                    # File is too small, regenerate
                    os.remove(line_audio_path)
                    existing_lines.discard(line_index)
            except Exception:
                pass
        
        try:
            if use_separate_dialogue:
                # Split line into dialogue and narration, generate separately and concatenate
                annotated_parts = split_and_annotate_text(line)
                combined_audio = AudioSegment.empty()
                
                for part in annotated_parts:
                    text_to_speak = part["text"].strip()
                    
                    if not text_to_speak or is_only_punctuation(text_to_speak):
                        continue
                    
                    # Use dialogue voice for dialogue, narrator voice for narration
                    voice_to_use = effective_dialogue if part["type"] == "dialogue" else effective_narrator
                    
                    # Remove quotes from text to speak
                    text_to_speak = text_to_speak.replace('"', '').replace('\\', '')
                    
                    if not text_to_speak or is_only_punctuation(text_to_speak):
                        continue
                    
                    # Create temporary file for this part
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    try:
                        audio_buffer = await generate_line_audio(
                            line_index, line, text_to_speak, voice_to_use, temp_path
                        )
                        
                        with open(temp_path, "wb") as temp_wav:
                            temp_wav.write(audio_buffer)
                        
                        part_segment = AudioSegment.from_wav(temp_path)
                        combined_audio += part_segment
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                # Check if we have any audio content
                if len(combined_audio) == 0:
                    progress_bar.update(1)
                    progress_counter += 1
                    return {"index": line_index, "is_chapter_heading": False, "line": line, "skipped": True}
                
                # Export combined audio
                combined_audio.export(line_audio_path, format="wav")
                
                # Mark line as complete in DB (source of truth for resume)
                if job_id:
                    job_manager.mark_line_complete(job_id, line_index)
            else:
                # Single voice mode - generate entire line as one TTS call for natural prosody
                text_to_speak = line.replace('"', '').replace('\\', '').strip()
                
                if not text_to_speak or is_only_punctuation(text_to_speak):
                    progress_bar.update(1)
                    progress_counter += 1
                    return {"index": line_index, "is_chapter_heading": False, "line": line, "skipped": True}
                
                audio_buffer = await generate_line_audio(
                    line_index, line, text_to_speak, effective_narrator, line_audio_path
                )
                
                with open(line_audio_path, "wb") as wav_file:
                    wav_file.write(audio_buffer)
            
            # Mark line as complete in DB (source of truth for resume)
            if job_id:
                job_manager.mark_line_complete(job_id, line_index)
            
            # Update progress bar
            progress_bar.update(1)
            progress_counter += 1
            
            # Save checkpoint periodically
            if job_id and (progress_counter - last_checkpoint_count) >= CHECKPOINT_INTERVAL:
                last_checkpoint_count = progress_counter
                checkpoint = JobCheckpoint(
                    total_lines=total_size,
                    lines_completed=progress_counter,
                    book_file_path=book_path,
                    add_emotion_tags=add_emotion_tags,
                    temp_audio_dir=temp_audio_dir
                )
                job_manager.update_job_checkpoint(job_id, checkpoint)
            
            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line,
            }
                
        except (RuntimeError, IOError, OSError) as e:
            # Log the error and mark as failed for retry
            error_msg = str(e)[:200]
            logger.error(f"Line {line_index} FAILED after {MAX_RETRIES_PER_LINE} attempts: '{line[:50]}...' - Error: {error_msg}")
            failed_lines[line_index] = (line, 1)
            progress_bar.update(1)
            progress_counter += 1
            return {"index": line_index, "line": line, "failed": True, "error": error_msg}

    # Process all lines with controlled concurrency
    yield JobProgress(f"Processing {total_size} lines with {TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE} parallel requests...", percent_complete=0, stage="generating")
    
    # Process in batches to avoid overwhelming the system
    BATCH_SIZE = TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE * 4  # Process 4x parallel limit at a time
    results_all = [None] * len(lines)
    
    for batch_start in range(0, len(lines), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(lines))
        batch_lines = [(i, lines[i]) for i in range(batch_start, batch_end)]
        
        # Create tasks for this batch
        tasks = []
        task_to_index = {}
        
        for line_index, line in batch_lines:
            # Skip if already processed successfully
            if line_index in existing_lines:
                line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.wav")
                if os.path.exists(line_audio_path):
                    # Validate file size - corrupted files are often very small
                    try:
                        file_size = os.path.getsize(line_audio_path)
                        if file_size > 1000:  # Valid WAV should be > 1KB
                            results_all[line_index] = {
                                "index": line_index,
                                "is_chapter_heading": check_if_chapter_heading(line),
                                "line": line,
                            }
                            continue
                        else:
                            # File is too small/corrupted, regenerate
                            logger.warning(f"⚠️ Existing file for line {line_index} is corrupted ({file_size} bytes), regenerating")
                            os.remove(line_audio_path)
                            existing_lines.discard(line_index)
                    except Exception as e:
                        logger.warning(f"⚠️ Error checking file for line {line_index}: {e}, regenerating")
                        existing_lines.discard(line_index)
            
            task = asyncio.create_task(process_single_line(line_index, line))
            tasks.append(task)
            task_to_index[task] = line_index
        
        # Process batch tasks
        if tasks:
            done_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task, result in zip(tasks, done_tasks):
                idx = task_to_index[task]
                if isinstance(result, Exception):
                    logger.error(f"❌ Task for line {idx} raised exception: {result}")
                    failed_lines[idx] = (lines[idx], 1)
                    results_all[idx] = {"index": idx, "line": lines[idx], "failed": True, "error": str(result)}
                else:
                    results_all[idx] = result
        
        # Progress update
        completed = sum(1 for r in results_all if r is not None)
        percent = (completed / total_size) * 100
        progress_msg = f"Generating audiobook. Progress: {percent:.1f}% ({completed}/{total_size})"
        logger.info(progress_msg)
        yield JobProgress(progress_msg, percent_complete=percent, stage="generating")
        
        # Save checkpoint after each batch
        if job_id:
            checkpoint = JobCheckpoint(
                total_lines=total_size,
                lines_completed=completed,
                book_file_path=book_path,
                add_emotion_tags=add_emotion_tags,
                temp_audio_dir=temp_audio_dir
            )
            job_manager.update_job_checkpoint(job_id, checkpoint)
    
    progress_bar.close()
    
    # CRITICAL: Retry any failed lines
    if failed_lines:
        yield JobProgress(f"⚠️ {len(failed_lines)} lines failed. Starting retry pass...", percent_complete=90, stage="retrying")
        
        retry_round = 1
        max_retry_rounds = 3
        
        while failed_lines and retry_round <= max_retry_rounds:
            yield JobProgress(f"Retry round {retry_round}/{max_retry_rounds} for {len(failed_lines)} failed lines...", percent_complete=90, stage="retrying")
            
            # Wait before retry
            await asyncio.sleep(5 * retry_round)
            
            lines_to_retry = list(failed_lines.items())
            failed_lines.clear()
            
            for line_index, (line_text, error_count) in lines_to_retry:
                try:
                    result = await process_single_line(line_index, line_text)
                    if result and not result.get("failed"):
                        results_all[line_index] = result
                        yield JobProgress(f"Line {line_index} succeeded on retry", percent_complete=90, stage="retrying")
                    else:
                        # Still failed
                        failed_lines[line_index] = (line_text, error_count + 1)
                except Exception as e:
                    logger.error(f"❌ Retry failed for line {line_index}: {e}")
                    failed_lines[line_index] = (line_text, error_count + 1)
            
            retry_round += 1
        
        if failed_lines:
            failed_count = len(failed_lines)
            yield JobProgress(f"WARNING: {failed_count} lines could not be generated after all retries!", percent_complete=90, stage="retrying", details={"failed_count": failed_count})
            for line_idx, (line_text, _) in list(failed_lines.items())[:5]:
                logger.error(f"  - Line {line_idx}: {line_text[:80]}...")
            
            # Save failed lines to file for manual inspection
            failed_lines_path = os.path.join(temp_audio_dir, "failed_lines.txt")
            with open(failed_lines_path, "w", encoding="utf-8") as f:
                for line_idx, (line_text, error_count) in failed_lines.items():
                    f.write(f"Line {line_idx}: {line_text}\n")
            yield f"📝 Failed lines saved to {failed_lines_path}"
    
    # Filter results - only include successful ones with verified audio files
    results = []
    missing_audio_count = 0
    for r in results_all:
        if r is None or r.get("failed") or r.get("skipped"):
            continue
        
        # CRITICAL: Verify the audio file actually exists before including in results
        line_audio_path = os.path.join(temp_line_audio_dir, f"line_{r['index']:06d}.wav")
        if not os.path.exists(line_audio_path):
            logger.warning(f"⚠️ Missing audio file for line {r['index']}: {r.get('line', '')[:50]}...")
            missing_audio_count += 1
            continue
            
        # Also verify file size
        try:
            file_size = os.path.getsize(line_audio_path)
            if file_size < 1000:
                print(f"⚠️ Corrupted audio file for line {r['index']} ({file_size} bytes), excluding from assembly")
                missing_audio_count += 1
                continue
        except Exception:
            missing_audio_count += 1
            continue
            
        results.append(r)
    
    if missing_audio_count > 0:
        yield JobProgress(f"{missing_audio_count} lines had missing or corrupted audio files", percent_complete=95, stage="filtering")
    
    yield JobProgress(f"Successfully generated audio for {len(results)}/{total_size} lines", percent_complete=95, stage="processing")

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(total=len(results), unit="result", desc="Organizing Chapters")
    
    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result.get("is_chapter_heading"):
            chapter_index += 1
            current_chapter_audio = f"{sanitize_filename(result['line'])}.wav"
            
        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []
            
        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)
    
    chapter_organization_bar.close()
    yield JobProgress("Organizing audio by chapters complete", percent_complete=98, stage="assembling")
    
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(total=len(chapter_files), unit="chapter", desc="Assembling Chapters")
    
    for chapter_file in chapter_files:
        # Use FFmpeg-based assembly instead of PyDub for memory efficiency
        assemble_chapter_with_ffmpeg(
            chapter_file, 
            chapter_line_map[chapter_file], 
            temp_line_audio_dir, 
            temp_audio_dir
        )
        
        chapter_assembly_bar.update(1)
        yield f"Assembled chapter: {chapter_file}"
    
    chapter_assembly_bar.close()
    yield "Completed assembling all chapters"

    # Enhanced Post-processing (Whole Book)
    if use_postprocessing:
        yield "✨ Starting enhanced post-processing for all chapters..."
        from audiobook.utils.audio_enhancer import audio_pipeline
        for idx, chapter_file in enumerate(chapter_files):
            chapter_path = os.path.join(temp_audio_dir, chapter_file)
            yield f"✨ Enhancing chapter {idx+1}/{len(chapter_files)}: {chapter_file}..."
            
            try:
                # Process in-place (pipeline will handle temp file and overwrite)
                # For whole book, we skip Demucs (TTS is clean) but use Enhancement and SR
                processed_path, status_msg = audio_pipeline.process(
                    chapter_path, 
                    output_path=chapter_path, # Overwrite with enhanced version
                    enable_preprocessing=True,
                    stages=['enhancement', 'sr']
                )
                if processed_path:
                    yield f"✅ Enhanced chapter {idx+1}: {status_msg}"
            except Exception as e:
                yield f"⚠️ Enhancement failed for {chapter_file}: {e}"
        
        yield "✨ Completed enhanced post-processing for all chapters"

    # Post-processing steps
    post_processing_bar = tqdm(total=len(chapter_files)*2, unit="task", desc="Post Processing")
    
    # Add silence to each chapter file using FFmpeg
    for chapter_file in chapter_files:
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        
        # Use FFmpeg-based silence addition instead of PyDub for memory efficiency
        add_silence_to_chapter_with_ffmpeg(chapter_path, 1000)  # 1 second silence
        
        post_processing_bar.update(1)
        yield f"Added silence to chapter: {chapter_file}"

    m4a_chapter_files = []

    # Convert all chapter files to M4A format
    for chapter_file in chapter_files:
        chapter_name = chapter_file.split('.')[0]
        m4a_chapter_files.append(f"{chapter_name}.m4a")
        # Convert WAV to M4A for better compatibility with timestamps and metadata
        convert_audio_file_formats("wav", "m4a", temp_audio_dir, chapter_name)
        post_processing_bar.update(1)
        yield f"Converted chapter to M4A: {chapter_name}"
    
    post_processing_bar.close()
    
    # Clean up temp line audio files
    shutil.rmtree(temp_line_audio_dir)
    yield "Cleaned up temporary files"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files, temp_audio_dir)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files, temp_audio_dir)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"


def apply_emotion_tags_to_multi_voice_data(json_data_array):
    """
    Dynamically apply pre-processed emotion tags to multi-voice JSONL data.
    
    This function reads emotion-enhanced text from tag_added_lines_chunks.txt
    and applies it to the speaker-attributed JSONL data in memory, preserving
    speaker attributions while using the enhanced text content.
    
    Args:
        json_data_array (list): Original speaker-attributed JSONL data
        
    Returns:
        tuple: (success, json_data_array, message)
            - success (bool): True if emotion tags were successfully applied
            - json_data_array (list): Updated JSONL data with emotion tags
            - message (str): Status message describing the result
    """
    if not os.path.exists("tag_added_lines_chunks.txt"):
        return False, json_data_array, "No pre-processed emotion tags found"
    
    try:
        # Read the enhanced lines from tag_added_lines_chunks.txt
        with open("tag_added_lines_chunks.txt", "r", encoding='utf-8') as f:
            enhanced_lines = f.read().split('\n')

        # Dynamically create enhanced JSONL data by matching enhanced lines with original speaker attributions
        if len(enhanced_lines) == len(json_data_array):
            for i, item in enumerate(json_data_array):
                item["line"] = enhanced_lines[i]
            return True, json_data_array, "Successfully applied pre-processed emotion tags"
        else:
            return False, json_data_array, f"Line count mismatch: {len(enhanced_lines)} enhanced lines vs {len(json_data_array)} speaker-attributed lines"
            
    except Exception as e:
        return False, json_data_array, f"Error applying emotion tags: {str(e)}"

async def generate_audio_with_multiple_voices(output_format, narrator_gender, generate_m4b_audiobook_file=False, book_path="", add_emotion_tags=False, use_postprocessing=False):
    # Path to the JSONL file containing speaker-attributed lines
    """
    Generate an audiobook in the specified format using multiple voices for each line

    Uses the provided JSONL file to map speaker names to voices. The JSONL file should contain
    entries with the following format:
    {
        "line": <string>,
        "speaker": <string>
    }

    The function will generate audio for each line independently and then concatenate the audio
    files for each chapter in order. The final audiobook will be saved in the "generated_audiobooks"
    directory with the name "audiobook.<format>".

    :param output_format: The desired format of the final audiobook (e.g. "m4a", "mp3")
    :param narrator_gender: The gender of the narrator voice (e.g. "male", "female")
    :param generate_m4b_audiobook_file: Whether to generate an M4B audiobook file instead of a standard
    M4A file
    :param book_path: The path to the book file (required for generating an M4B audiobook file)
    :param add_emotion_tags: Whether to use pre-applied emotion tags in the audiobook. Defaults to False.
    :param use_postprocessing: Whether to run audio enhancement on each chapter. Defaults to False.
    """
    
    # Early validation for M4B generation
    if generate_m4b_audiobook_file:
        yield "Validating book file for M4B audiobook generation..."
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_path)
        
        if not is_valid:
            raise ValueError(f"❌ Book validation failed: {error_message}")
            
        yield f"✅ Book validation successful! Title: {metadata.get('Title', 'Unknown')}, Author: {metadata.get('Author(s)', 'Unknown')}"
    
    file_path = 'speaker_attributed_book.jsonl'
    json_data_array = []

    # Open the JSONL file and read it line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())
            # Append the parsed JSON object to the array
            json_data_array.append(json_object)

    yield "Loaded speaker-attributed lines from JSONL file"

    # Apply emotion tags if requested and available
    if add_emotion_tags:
        success, json_data_array, message = apply_emotion_tags_to_multi_voice_data(json_data_array)
        if success:
            yield f"✅ {message}"
        else:
            yield f"⚠️ {message}"
            yield "Falling back to original text without emotion tags"
    else:
        # Check if emotion tags exist in the original JSONL data and remove them if user doesn't want them
        has_emotion_tags = any(
            '<laugh>' in item.get('line', '') or '<chuckle>' in item.get('line', '') or
            '<sigh>' in item.get('line', '') or '<cough>' in item.get('line', '') or
            '<sniffle>' in item.get('line', '') or '<groan>' in item.get('line', '') or
            '<yawn>' in item.get('line', '') or '<gasp>' in item.get('line', '')
            for item in json_data_array
        )
        
        if has_emotion_tags:
            yield "Removing existing emotion tags from JSONL data as per user preference"
            import re
            for item in json_data_array:
                if "line" in item and item["line"]:
                    # Remove emotion tags from the line
                    line_without_tags = re.sub(r'<(?:laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*', '', item["line"])
                    item["line"] = line_without_tags
    
    # Apply text preprocessing for Orpheus TTS to prevent repetition issues
    if TTS_MODEL.lower() == "orpheus":
        for item in json_data_array:
            if "line" in item and item["line"]:
                item["line"] = preprocess_text_for_tts(item["line"])
        yield "Applied text preprocessing for Orpheus TTS"

    # Load mappings for character gender
    character_gender_map = read_json("character_gender_map.json")

    # Get narrator voice using the new voice mapping system
    narrator_voice = find_voice_for_gender_score("narrator", character_gender_map, TTS_MODEL, narrator_gender)
    yield "Loaded voice mappings and selected narrator voice"
    
    # Setup directories
    temp_audio_dir = "temp_audio"
    temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")

    empty_directory(temp_audio_dir)

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    yield "Set up temporary directories for audio processing"
    
    # Batch processing parameters
    semaphore = asyncio.Semaphore(TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.wav"
    chapter_files = []
    
    # First pass: Generate audio for each line independently
    # and track chapter organization
    chapter_line_map = {}  # Maps chapters to their line indices

    progress_counter = 0
    
    # For tracking progress with tqdm in an async context
    total_lines = len(json_data_array)
    progress_bar = tqdm(total=total_lines, unit="line", desc="Audio Generation Progress")

    yield "Generating audio..."

    async def process_single_line(line_index, doc):
        async with semaphore:
            nonlocal progress_counter

            line = doc["line"].strip()

            if not line or is_only_punctuation(line):
                progress_bar.update(1)
                progress_counter += 1
                return None

            speaker = doc["speaker"]
            speaker_voice = find_voice_for_gender_score(speaker, character_gender_map, TTS_MODEL, narrator_gender)
            
            # Split the line into annotated parts
            annotated_parts = split_and_annotate_text(line)
            
            # Create combined audio using PyDub for seamless concatenation
            combined_audio = AudioSegment.empty()
            
            for part in annotated_parts:
                text_to_speak = part["text"].strip()

                if not text_to_speak or is_only_punctuation(text_to_speak):
                    continue

                voice_to_speak_in = narrator_voice if part["type"] == "narration" else speaker_voice

                # strip all double quotes and backslashes from the text to speak
                text_to_speak = text_to_speak.replace('"', '').replace('\\', '')
                
                # Create temporary file for this part
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_file.name
                temp_file.close()
                
                try:
                    # Generate audio for the part using retry mechanism
                    audio_buffer = await generate_audio_with_retry(
                        async_openai_client, 
                        TTS_MODEL,
                        text_to_speak, 
                        voice_to_speak_in
                    )
                    
                    # Write part audio to temp file
                    with open(temp_path, "wb") as temp_wav:
                        temp_wav.write(audio_buffer)
                    
                    # Load as AudioSegment and add to combined audio
                    part_segment = AudioSegment.from_wav(temp_path)
                    combined_audio += part_segment
                    
                except Exception as e:
                    # Log the error for debugging
                    print(f"Warning: Failed to generate audio for text: '{text_to_speak[:50]}...' - Error: {str(e)}")
                    # Skip this part and continue with next part
                    
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Check if we have any audio content before exporting
            if len(combined_audio) == 0:
                # If no audio was generated for this line, skip it entirely
                progress_bar.update(1)
                progress_counter += 1
                return None
            
            # Write this line's audio to a temporary file
            line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.wav")
            combined_audio.export(line_audio_path, format="wav")
            
            # Update progress bar
            progress_bar.update(1)
            progress_counter += 1
            
            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line
            }
    
    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, doc in enumerate(json_data_array):
        task = asyncio.create_task(process_single_line(i, doc))
        tasks.append(task)
        task_to_index[task] = i
    
    # Initialize results_all list
    results_all = [None] * len(json_data_array)
    
    # Process tasks with progress updates
    last_reported = -1
    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Store results as tasks complete
        for completed_task in done:
            idx = task_to_index[completed_task]
            results_all[idx] = completed_task.result()
        
        tasks = list(pending)
        
        # Only yield if the counter has changed
        if progress_counter > last_reported:
            last_reported = progress_counter
            percent = (progress_counter / total_lines) * 100
            yield f"Generating audiobook. Progress: {percent:.1f}%"
    
    # All tasks have completed at this point and results_all is populated
    progress_bar.close()
    
    # Filter results - only include successful ones with verified audio files
    results = []
    missing_audio_count = 0
    for r in results_all:
        if r is None:
            continue
        
        # CRITICAL: Verify the audio file actually exists before including in results
        line_audio_path = os.path.join(temp_line_audio_dir, f"line_{r['index']:06d}.wav")
        if not os.path.exists(line_audio_path):
            print(f"⚠️ Missing audio file for line {r['index']}: {r.get('line', '')[:50]}...")
            missing_audio_count += 1
            continue
            
        # Also verify file size
        try:
            file_size = os.path.getsize(line_audio_path)
            if file_size < 1000:
                print(f"⚠️ Corrupted audio file for line {r['index']} ({file_size} bytes), excluding from assembly")
                missing_audio_count += 1
                continue
        except Exception:
            missing_audio_count += 1
            continue
            
        results.append(r)
    
    if missing_audio_count > 0:
        yield f"⚠️ {missing_audio_count} lines had missing or corrupted audio files"
    
    yield "Completed generating audio for all lines"
    
    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(total=len(results), unit="result", desc="Organizing Chapters")
    yield "Organizing lines into chapters"
    
    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1
            current_chapter_audio = f"{sanitize_filename(result['line'])}.wav"
            
        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []
            
        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)
    
    chapter_organization_bar.close()
    yield f"Organized {len(results)} lines into {len(chapter_files)} chapters"
    
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(total=len(chapter_files), unit="chapter", desc="Assembling Chapters")
    
    for chapter_file in chapter_files:
        # Use FFmpeg-based assembly instead of PyDub for memory efficiency
        assemble_chapter_with_ffmpeg(
            chapter_file, 
            chapter_line_map[chapter_file], 
            temp_line_audio_dir, 
            temp_audio_dir
        )
        
        chapter_assembly_bar.update(1)
        yield f"Assembled chapter: {chapter_file}"
    
    chapter_assembly_bar.close()
    yield "Completed assembling all chapters"
    
    # Enhanced Post-processing (Whole Book)
    if use_postprocessing:
        yield "✨ Starting enhanced post-processing for all chapters..."
        from audiobook.utils.audio_enhancer import audio_pipeline
        for idx, chapter_file in enumerate(chapter_files):
            chapter_path = os.path.join(temp_audio_dir, chapter_file)
            yield f"✨ Enhancing chapter {idx+1}/{len(chapter_files)}: {chapter_file}..."
            
            try:
                # Process in-place (pipeline will handle temp file and overwrite)
                # For whole book, we skip Demucs (TTS is clean) but use Enhancement and SR
                processed_path, status_msg = audio_pipeline.process(
                    chapter_path, 
                    output_path=chapter_path, # Overwrite with enhanced version
                    enable_preprocessing=True,
                    stages=['enhancement', 'sr']
                )
                if processed_path:
                    yield f"✅ Enhanced chapter {idx+1}: {status_msg}"
            except Exception as e:
                yield f"⚠️ Enhancement failed for {chapter_file}: {e}"
        
        yield "✨ Completed enhanced post-processing for all chapters"

    # Post-processing steps
    post_processing_bar = tqdm(total=len(chapter_files)*2, unit="task", desc="Post Processing")
    
    # Add silence to each chapter file using FFmpeg
    for chapter_file in chapter_files:
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        
        # Use FFmpeg-based silence addition instead of PyDub for memory efficiency
        add_silence_to_chapter_with_ffmpeg(chapter_path, 1000)  # 1 second silence
        
        post_processing_bar.update(1)
        yield f"Added silence to chapter: {chapter_file}"

    m4a_chapter_files = []

    # Convert all chapter files to M4A format
    for chapter_file in chapter_files:
        chapter_name = chapter_file.split('.')[0]
        m4a_chapter_files.append(f"{chapter_name}.m4a")
        # Convert WAV to M4A for better compatibility with timestamps and metadata
        convert_audio_file_formats("wav", "m4a", temp_audio_dir, chapter_name)
        post_processing_bar.update(1)
        yield f"Converted chapter to M4A: {chapter_name}"
    
    post_processing_bar.close()
    
    # Clean up temp line audio files
    yield "Cleaning up temporary files"
    shutil.rmtree(temp_line_audio_dir)
    yield "Temporary files cleanup complete"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files, temp_audio_dir)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files, temp_audio_dir)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"




async def main():
    os.makedirs("generated_audiobooks", exist_ok=True)

    # Default values
    book_path = None
    generate_m4b_audiobook_file = False
    output_format = "aac"

    # Prompt user for audiobook type selection
    print("\n🎙️ **Audiobook Type Selection**")
    print("🔹 Do you want the audiobook in M4B format (the standard format for audiobooks) with chapter timestamps and embedded book cover ? (Needs calibre and ffmpeg installed)")
    print("🔹 OR do you want a standard audio file in either of ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm'] formats without any of the above features ?")
    audiobook_type_option = input("🔹 Enter **1** for **M4B audiobook format** or **2** for **Standard Audio File**: ").strip()

    if audiobook_type_option == "1":
        is_calibre_installed = check_if_calibre_is_installed()

        if not is_calibre_installed:
            print("⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-meta** commands are available in your PATH.")
            return
        
        is_ffmpeg_installed = check_if_ffmpeg_is_installed()

        if not is_ffmpeg_installed:
            print("⚠️ FFMpeg is not installed. Please install it first and make sure **ffmpeg** and **ffprobe** commands are available in your PATH.")
            return

        # Check if a path is provided via command-line arguments
        if len(sys.argv) > 1:
            book_path = sys.argv[1]
            print(f"📂 Using book file from command-line argument: **{book_path}**")
        else:
            # Ask user for book file path if not provided
            input_path = input("\n📖 Enter the **path to the book file**, needed for metadata and cover extraction. (Press Enter to use default): ").strip()
            if input_path:
                book_path = input_path
            print(f"📂 Using book file: **{book_path}**")

        print("✅ Book path set. Proceeding...\n")
        
        # Early validation of the book file for M4B generation
        print("🔍 Validating book file for M4B audiobook generation...")
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_path)
        
        if not is_valid:
            print(f"❌ **Book validation failed**: {error_message}")
            print("\n💡 **Troubleshooting Tips:**")
            print("   • Ensure the book file path is correct and the file exists")
            print("   • Verify the book file is a supported ebook format (EPUB, MOBI, PDF, etc.)")
            print("   • Check that Calibre is properly installed and ebook-meta command is available")
            print("   • Make sure the book file is not corrupted")
            print("   • Ensure the book file contains extractable metadata and cover image")
            return
            
        print(f"✅ **Book validation successful!**")
        print(f"   • Title: {metadata.get('Title', 'Unknown')}")
        print(f"   • Author: {metadata.get('Author(s)', 'Unknown')}")
        print(f"   • Cover image: Successfully extracted")
        print()

        generate_m4b_audiobook_file = True
    else:
        # Prompt user for audio format selection
        print("\n🎙️ **Audiobook Output Format Selection**")
        output_format = input("🔹 Choose between ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm']. ").strip()

        if(output_format not in ["aac", "m4a", "mp3", "wav", "opus", "flac", "pcm"]):
            print("\n⚠️ Invalid output format! Please choose from the give options")
            return
        
    # Prompt user for narrator's gender selection
    print("\n🎙️ **Audiobook Narrator Voice Selection**")
    narrator_gender = input("🔹 Enter **male** if you want the book to be read in a male voice or **female** if you want the book to be read in a female voice: ").strip()

    if narrator_gender not in ["male", "female"]:
        print("\n⚠️ Invalid narrator gender! Please choose from the give options")
        return

    # Prompt user for emotion tags option (always available with Orpheus TTS)
    add_emotion_tags = False
    print("\n🎭 **Emotion Tags Enhancement (Orpheus TTS)**")
    print("🔹 Emotion tags add natural expressions like laughter, sighs, gasps to your audiobook")
    print("🔹 Available tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")
    emotion_tags_option = input("🔹 Do you want to use emotion tags in the audiobook? Enter **yes** or **no**: ").strip().lower()
    
    if emotion_tags_option in ["yes", "y", "true", "1"]:
        add_emotion_tags = True
        print("✅ Emotion tags will be used in the audiobook!")
    else:
        print("ℹ️ Emotion tags disabled. Standard narration will be used.")

    start_time = time.time()

    print("\n🎧 Generating audiobook with Orpheus TTS...")
    async for line in generate_audio_with_single_voice(output_format, narrator_gender, generate_m4b_audiobook_file, book_path, add_emotion_tags):
        print(line)

    print(f"\n🎧 Audiobook is generated! The audiobook is saved as **audiobook.{'m4b' if generate_m4b_audiobook_file else output_format}** in the **generated_audiobooks** directory in the current folder.")

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds\n✅ Audiobook generation complete!")


async def process_audiobook_generation_v2(
    job_id: str,
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
):
    """
    Process audiobook generation using the new pipeline architecture.
    
    This is the preferred entry point for audiobook generation. It uses:
    - TextLoader for text loading
    - LineGenerator for TTS generation
    - ChapterAssembler for audio assembly
    - AudiobookPipeline to orchestrate everything
    
    Args:
        job_id: Job identifier for state tracking
        book_path: Path to the book file
        engine: TTS engine to use (orpheus, chatterbox, vibevoice)
        voice: Voice for narration
        output_format: Output format (m4b, mp3, m4a, etc.)
        add_emotion_tags: Whether to use emotion tags
        dialogue_voice: Optional separate voice for dialogue
        use_postprocessing: Whether to apply audio enhancement
        reference_audio_path: For voice cloning (Chatterbox)
        vibevoice_temperature: Temperature for VibeVoice
        vibevoice_top_p: Top-p for VibeVoice
        
    Yields:
        JobProgress updates throughout generation.
    """
    from audiobook.tts.pipeline import AudiobookPipeline, PipelineConfig
    
    # Normalize output format
    fmt = output_format.lower()
    if "m4b" in fmt or "chapters" in fmt.lower():
        fmt = "m4b"
    
    config = PipelineConfig(
        job_id=job_id,
        book_path=book_path,
        output_format=fmt,
        engine=engine.lower(),
        voice=voice,
        dialogue_voice=dialogue_voice,
        use_separate_dialogue=bool(dialogue_voice),
        add_emotion_tags=add_emotion_tags,
        generate_m4b=(fmt == "m4b"),
        use_postprocessing=use_postprocessing,
        reference_audio_path=reference_audio_path,
        vibevoice_temperature=vibevoice_temperature,
        vibevoice_top_p=vibevoice_top_p,
    )
    
    pipeline = AudiobookPipeline(config)
    async for progress in pipeline.run():
        yield progress


if __name__ == "__main__":
    asyncio.run(main())