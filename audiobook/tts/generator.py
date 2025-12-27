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
from audiobook.utils.llm_utils import generate_audio_with_retry
from dotenv import load_dotenv

load_dotenv()

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1")
TTS_API_KEY = os.environ.get("TTS_API_KEY", "not-needed")
TTS_MODEL = "orpheus"  # Always use Orpheus TTS
TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE = int(os.environ.get("TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE", 4))

os.makedirs("audio_samples", exist_ok=True)
os.makedirs("generated_audiobooks", exist_ok=True)

async_openai_client = AsyncOpenAI(
    base_url=TTS_BASE_URL, api_key=TTS_API_KEY
)

def sanitize_filename(text):
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

async def generate_audio_with_single_voice(output_format, narrator_voice, generate_m4b_audiobook_file=False, book_path="", add_emotion_tags=False, job_id=None, resume_checkpoint=None):
    # Read the text from the file
    """
    Generate an audiobook using a single voice for narration and dialogues.

    This asynchronous function reads text from a file, processes each line to determine
    if it is narration or dialogue, and generates corresponding audio using specified
    voices. The generated audio is organized by chapters, with options to create
    an M4B audiobook file or a standard audio file in the specified output format.

    Args:
        output_format (str): The desired output format for the final audiobook (e.g., "mp3", "wav").
        narrator_voice (str): The voice name to use for narration (e.g., "zac", "tara").
        generate_m4b_audiobook_file (bool, optional): Flag to determine whether to generate an M4B file. Defaults to False.
        book_path (str, optional): The file path for the book to be used in M4B creation. Defaults to an empty string.
        add_emotion_tags (bool, optional): Whether to use pre-applied emotion tags in the audiobook. Defaults to False.
        job_id (str, optional): Job ID for checkpoint saving. Defaults to None.
        resume_checkpoint (dict, optional): Checkpoint data for resuming a job. Defaults to None.

    Yields:
        str: Progress updates as the audiobook generation progresses through loading text, generating audio,
             organizing by chapters, assembling chapters, and post-processing steps.
    """
    # Import job manager for checkpoint saving (only if job_id provided)
    if job_id:
        from audiobook.utils.job_manager import job_manager, JobCheckpoint
    
    # Determine if we're resuming
    is_resuming = resume_checkpoint is not None
    
    # Early validation for M4B generation (skip if resuming - already validated)
    if generate_m4b_audiobook_file and not is_resuming:
        yield "Validating book file for M4B audiobook generation..."
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_path)
        
        if not is_valid:
            raise ValueError(f"❌ Book validation failed: {error_message}")
            
        yield f"✅ Book validation successful! Title: {metadata.get('Title', 'Unknown')}, Author: {metadata.get('Author(s)', 'Unknown')}"
        
        # Copy book file to persistent location for resume capability
        if book_path and os.path.exists(book_path):
            import shutil
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
    if add_emotion_tags and os.path.exists("tag_added_lines_chunks.txt"):
        with open("tag_added_lines_chunks.txt", "r", encoding='utf-8') as f:
            text = f.read()
        yield "Using pre-processed text with emotion tags"
    else:
        with open("converted_book.txt", "r", encoding='utf-8') as f:
            text = f.read()
        
        # Apply text preprocessing for Orpheus TTS to prevent repetition issues
        if TTS_MODEL.lower() == "orpheus":
            text = preprocess_text_for_tts(text)
            yield "Applied text preprocessing for Orpheus TTS"
    
    lines = text.split("\n")
    
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Use the provided narrator voice directly, dialogue voice same as narrator
    dialogue_voice = narrator_voice
    yield f"Using voice: {narrator_voice}"

    # Setup directories
    temp_audio_dir = "temp_audio"
    temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")

    # Only clear directories if NOT resuming
    if not resume_checkpoint:
        empty_directory(temp_audio_dir)

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    
    # Batch processing parameters
    semaphore = asyncio.Semaphore(TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.wav"
    chapter_files = []
    
    # First pass: Generate audio for each line independently
    total_size = len(lines)

    progress_counter = 0
    
    # Check for existing line audio files (for resume support)
    existing_lines = set()
    if resume_checkpoint or os.path.exists(temp_line_audio_dir):
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
    
    async def process_single_line(line_index, line):
        async with semaphore:
            nonlocal progress_counter, last_checkpoint_count

            if not line or is_only_punctuation(line):
                progress_bar.update(1)
                progress_counter += 1
                return None
            
            # Check if this line was already generated (resume support)
            line_audio_path = os.path.join(temp_line_audio_dir, f"line_{line_index:06d}.wav")
            if line_index in existing_lines and os.path.exists(line_audio_path):
                # Line already generated, skip
                progress_bar.update(1)
                progress_counter += 1
                return {
                    "index": line_index,
                    "is_chapter_heading": check_if_chapter_heading(line),
                    "line": line,
                }
                
            # Split the line into annotated parts
            annotated_parts = split_and_annotate_text(line)
            
            # Create combined audio using PyDub for seamless concatenation
            combined_audio = AudioSegment.empty()
            
            for part in annotated_parts:
                text_to_speak = part["text"].strip()

                if not text_to_speak or is_only_punctuation(text_to_speak):
                    continue

                voice_to_speak_in = narrator_voice if part["type"] == "narration" else dialogue_voice

                # strip all double quotes from the text to speak
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
            combined_audio.export(line_audio_path, format="wav")
            
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

    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, line in enumerate(lines):
        task = asyncio.create_task(process_single_line(i, line))
        tasks.append(task)
        task_to_index[task] = i
    
    # Initialize results_all list
    results_all = [None] * len(lines)
    
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
            percent = (progress_counter / total_size) * 100
            yield f"Generating audiobook. Progress: {percent:.1f}%"
    
    # All tasks have completed at this point and results_all is populated
    results = [r for r in results_all if r is not None]  # Filter out empty lines
    
    progress_bar.close()
    
    # Filter out empty lines (same as in your original code)
    results = [r for r in results_all if r is not None]
    
    yield "Completed generating audio for all lines"

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(total=len(results), unit="result", desc="Organizing Chapters")
    
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
    yield "Organizing audio by chapters complete"
    
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
        merge_chapters_to_m4b(book_path, m4a_chapter_files)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"


async def generate_audio_with_chatterbox(output_format, reference_audio_path, generate_m4b_audiobook_file=False, book_path=""):
    """
    Generate an audiobook using Chatterbox TTS with zero-shot voice cloning.

    This function uses a reference audio sample to clone a voice and generate
    the entire audiobook in that cloned voice.

    Args:
        output_format (str): The desired output format for the final audiobook (e.g., "mp3", "wav").
        reference_audio_path (str): Path to the reference audio file for voice cloning.
        generate_m4b_audiobook_file (bool, optional): Flag to determine whether to generate an M4B file. Defaults to False.
        book_path (str, optional): The file path for the book to be used in M4B creation. Defaults to an empty string.

    Yields:
        str: Progress updates as the audiobook generation progresses.
    """
    import torch
    import torchaudio as ta
    
    # Early validation for M4B generation
    if generate_m4b_audiobook_file:
        yield "Validating book file for M4B audiobook generation..."
        is_valid, error_message, metadata = validate_book_for_m4b_generation(book_path)
        
        if not is_valid:
            raise ValueError(f"❌ Book validation failed: {error_message}")
            
        yield f"✅ Book validation successful! Title: {metadata.get('Title', 'Unknown')}, Author: {metadata.get('Author(s)', 'Unknown')}"

    # Validate reference audio
    if not reference_audio_path or not os.path.exists(reference_audio_path):
        raise ValueError("Reference audio file is required for Chatterbox voice cloning")
    
    yield "⏳ Loading Chatterbox TTS model..."
    
    # Import and initialize Chatterbox (zero-shot voice cloning)
    try:
        from chatterbox import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        yield f"✅ Chatterbox model loaded on {device.upper()}"
    except ImportError:
        raise ImportError("❌ Chatterbox is not installed. Run: pip install chatterbox-tts")
    except Exception as e:
        raise RuntimeError(f"Failed to load Chatterbox model: {str(e)}")

    # Pre-compute voice conditioning from reference audio ONCE
    # This avoids re-processing the reference audio for every line
    yield "🎤 Extracting voice characteristics from reference audio..."
    try:
        chatterbox_model.prepare_conditionals(reference_audio_path, exaggeration=0.5)
        yield "✅ Voice conditioning cached - will be reused for all lines"
    except Exception as e:
        raise RuntimeError(f"Failed to extract voice from reference audio: {str(e)}")

    # Read the text
    with open("converted_book.txt", "r", encoding='utf-8') as f:
        text = f.read()
    
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    
    yield f"📖 Found {len(lines)} lines to process"

    # Setup directories
    temp_audio_dir = "temp_audio"
    temp_line_audio_dir = os.path.join(temp_audio_dir, "line_segments")

    empty_directory(temp_audio_dir)

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    
    # Initial setup for chapters
    chapter_index = 1
    current_chapter_audio = f"Introduction.wav"
    chapter_files = []
    
    total_size = len(lines)
    progress_bar = tqdm(total=total_size, unit="line", desc="Chatterbox Audio Generation")
    
    # Maps chapters to their line indices
    chapter_line_map = {}
    
    # Time tracking for ETA estimation
    import time
    generation_start_time = time.time()
    lines_generated = 0
    
    def format_time(seconds):
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def process_single_line_chatterbox(line_index, line):
        """Process a single line with Chatterbox TTS using pre-cached voice conditioning."""
        nonlocal lines_generated
        
        if not line or is_only_punctuation(line):
            progress_bar.update(1)
            return None
            
        # Clean up the text
        text_to_speak = line.replace('"', '').replace('\\', '').strip()
        
        if not text_to_speak or is_only_punctuation(text_to_speak):
            progress_bar.update(1)
            return None
        
        try:
            # Generate audio with Chatterbox using PRE-CACHED voice conditioning
            # No audio_prompt_path = uses the cached self.conds from prepare_conditionals
            wav = chatterbox_model.generate(text_to_speak)
            
            # Save to temp file
            output_path = os.path.join(temp_line_audio_dir, f"line_{line_index:05d}.wav")
            ta.save(output_path, wav, chatterbox_model.sr)
            
            progress_bar.update(1)
            lines_generated += 1
            return {
                "line_index": line_index,
                "audio_path": output_path
            }
            
        except Exception as e:
            print(f"Warning: Failed to generate audio for line {line_index}: '{text_to_speak[:50]}...' - Error: {str(e)}")
            progress_bar.update(1)
            return None
    
    # Process lines (Chatterbox is synchronous, so we process sequentially)
    line_audio_results = []
    
    for line_index, line in enumerate(lines):
        # Check for chapter headings
        if check_if_chapter_heading(line):
            chapter_line_map[chapter_index] = line_index
            chapter_index += 1
        
        result = process_single_line_chatterbox(line_index, line)
        if result:
            line_audio_results.append(result)
        
        # Calculate progress and ETA
        progress_pct = ((line_index + 1) / total_size) * 100
        elapsed = time.time() - generation_start_time
        
        # Calculate ETA based on lines actually generated (not skipped)
        if lines_generated > 0:
            avg_time_per_line = elapsed / lines_generated
            remaining_lines = total_size - (line_index + 1)
            # Estimate that ~70% of remaining lines will need generation
            estimated_remaining = remaining_lines * avg_time_per_line * 0.7
            eta_str = format_time(estimated_remaining)
            elapsed_str = format_time(elapsed)
            yield f"🎙️ Generating: {line_index + 1}/{total_size} ({progress_pct:.1f}%) | ⏱️ {elapsed_str} elapsed | ⏳ ~{eta_str} remaining"
        else:
            yield f"🎙️ Generating audio: {line_index + 1}/{total_size} lines ({progress_pct:.1f}%)"
    
    total_time = time.time() - generation_start_time
    progress_bar.close()
    yield f"✅ Generated {len(line_audio_results)} audio segments in {format_time(total_time)}"
    
    # Sort results by line index
    line_audio_results.sort(key=lambda x: x["line_index"])
    
    # Organize by chapters
    yield "Organizing audio by chapters..."
    
    chapter_boundaries = sorted(chapter_line_map.values())
    
    if not chapter_boundaries:
        # No chapters found, everything goes into Introduction
        chapter_line_ranges = [("Introduction", 0, len(lines))]
    else:
        chapter_line_ranges = []
        # Introduction (before first chapter)
        if chapter_boundaries[0] > 0:
            chapter_line_ranges.append(("Introduction", 0, chapter_boundaries[0]))
        
        # Each chapter
        for i, boundary in enumerate(chapter_boundaries):
            chapter_num = list(chapter_line_map.keys())[list(chapter_line_map.values()).index(boundary)]
            end_boundary = chapter_boundaries[i + 1] if i + 1 < len(chapter_boundaries) else len(lines)
            chapter_line_ranges.append((f"Chapter_{chapter_num}", boundary, end_boundary))
    
    # Assemble chapters
    yield "Assembling chapters..."
    
    total_chapters = len(chapter_line_ranges)
    for chapter_idx, (chapter_name, start_idx, end_idx) in enumerate(chapter_line_ranges):
        chapter_audio_files = []
        
        for result in line_audio_results:
            if start_idx <= result["line_index"] < end_idx:
                chapter_audio_files.append(result["audio_path"])
        
        if chapter_audio_files:
            chapter_output = os.path.join(temp_audio_dir, f"{chapter_name}.wav")
            
            # Use FFmpeg to concatenate audio files
            concatenate_audio_files_with_ffmpeg(chapter_audio_files, chapter_output)
            chapter_files.append(f"{chapter_name}.wav")
            yield f"📚 Assembling chapters: {chapter_idx + 1}/{total_chapters} ({chapter_name})"
    
    # Post-processing
    yield "⏳ Post-processing chapters..."
    
    # Add silence to each chapter file
    for idx, chapter_file in enumerate(chapter_files):
        chapter_path = os.path.join(temp_audio_dir, chapter_file)
        add_silence_to_chapter_with_ffmpeg(chapter_path, 1000)

    m4a_chapter_files = []

    # Convert all chapter files to M4A format
    yield "🔄 Converting chapters to M4A format..."
    for idx, chapter_file in enumerate(chapter_files):
        chapter_name = chapter_file.split('.')[0]
        m4a_chapter_files.append(f"{chapter_name}.m4a")
        convert_audio_file_formats("wav", "m4a", temp_audio_dir, chapter_name)
        yield f"🔄 Converting: {idx + 1}/{len(chapter_files)} chapters"
    
    # Clean up temp line audio files
    shutil.rmtree(temp_line_audio_dir)
    yield "🧹 Cleaned up temporary files"

    if generate_m4b_audiobook_file:
        yield "📀 Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files)
        yield "✅ M4B audiobook created successfully!"
    else:
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files)
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

async def generate_audio_with_multiple_voices(output_format, narrator_gender, generate_m4b_audiobook_file=False, book_path="", add_emotion_tags=False):
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
    results = [r for r in results_all if r is not None]  # Filter out empty lines
    
    progress_bar.close()
    
    # Filter out empty lines (same as in your original code)
    results = [r for r in results_all if r is not None]
    
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
        merge_chapters_to_m4b(book_path, m4a_chapter_files)
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files)
        convert_audio_file_formats("m4a", output_format, "generated_audiobooks", "audiobook")
        yield f"Audiobook in {output_format} format created successfully"

async def process_audiobook_generation(voice_option, narrator_voice, output_format, book_path, add_emotion_tags=False, tts_engine="Orpheus", reference_audio_path=None, job_id=None, resume_checkpoint=None):
    """
    Process audiobook generation with single voice narration.
    
    Args:
        voice_option: Ignored (single voice only)
        narrator_voice: Voice name to use for narration (e.g., "zac", "tara") - Orpheus only
        output_format: Output audio format
        book_path: Path to source book for M4B metadata
        add_emotion_tags: Whether to use pre-processed emotion tags - Orpheus only
        tts_engine: TTS engine to use ("Orpheus" or "Chatterbox")
        reference_audio_path: Path to reference audio for zero-shot cloning - Chatterbox only
        job_id: Job ID for checkpoint saving and resume support
        resume_checkpoint: Checkpoint data for resuming a stalled job
    """
    generate_m4b_audiobook_file = False

    if output_format == "M4B (Chapters & Cover)":
        generate_m4b_audiobook_file = True

    try:
        if tts_engine == "Chatterbox":
            # Chatterbox TTS - zero-shot voice cloning
            if not reference_audio_path:
                raise ValueError("Reference audio file is required for Chatterbox TTS")
            
            yield "\n🎧 Generating audiobook with Chatterbox TTS (Zero-Shot Voice Cloning)..."
            await asyncio.sleep(1)
            async for line in generate_audio_with_chatterbox(output_format.lower(), reference_audio_path, generate_m4b_audiobook_file, book_path):
                yield line
        else:
            # Orpheus TTS - predefined voices
            is_audio_generator_api_up, message = await check_if_audio_generator_api_is_up(async_openai_client)

            if not is_audio_generator_api_up:
                raise Exception(message)
            
            if resume_checkpoint:
                yield "\n🔄 Resuming audiobook generation with Orpheus TTS..."
            else:
                yield "\n🎧 Generating audiobook with Orpheus TTS..."
            await asyncio.sleep(1)
            async for line in generate_audio_with_single_voice(
                output_format.lower(), 
                narrator_voice, 
                generate_m4b_audiobook_file, 
                book_path, 
                add_emotion_tags,
                job_id=job_id,
                resume_checkpoint=resume_checkpoint
            ):
                yield line

        yield f"\n🎧 Audiobook is generated! You can now download it in the Download section below."
        
    except ValueError as e:
        # Handle validation errors specifically
        error_msg = str(e)
        if "Book validation failed" in error_msg:
            yield f"\n❌ **Book Validation Error**: {error_msg}"
            yield "\n💡 **Troubleshooting Tips:**"
            yield "   • Ensure the book file path is correct and the file exists"
            yield "   • Verify the book file is a supported ebook format (EPUB, MOBI, PDF, etc.)"
            yield "   • Check that Calibre is properly installed and ebook-meta command is available"
            yield "   • Make sure the book file is not corrupted"
            yield "   • Ensure the book file contains extractable metadata and cover image"
        else:
            yield f"\n❌ **Validation Error**: {error_msg}"
        raise e
    except Exception as e:
        yield f"\n❌ **Unexpected Error**: {str(e)}"
        raise e

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

if __name__ == "__main__":
    asyncio.run(main())