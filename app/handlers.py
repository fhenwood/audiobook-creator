"""
Request handlers for Audiobook Creator.

Contains wrapper functions for book validation, text extraction, and voice sample generation.
Extracted from app.py for modularity.
"""

import os
import traceback
import gradio as gr
from audiobook.core.text_extraction import process_book_and_extract_text, save_book
from audiobook.tts.service import tts_service
from audiobook.tts.generator import sanitize_filename


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


async def generate_voice_sample(
    engine_id, 
    voice_id, 
    sample_text, 
    reference_audio=None, 
    vibevoice_voice=None, 
    use_postprocessing=False, 
    vibevoice_temperature=0.7, 
    vibevoice_top_p=0.95
):
    """Generate a voice sample using the selected engine."""
    if not sample_text or not sample_text.strip():
        return None, "Please enter some text to generate a sample."
    
    try:
        engine_id = engine_id.lower()
        
        # Determine voice ID based on engine
        if engine_id == "orpheus":
            if not voice_id: 
                return None, "Please select an Orpheus voice."
        elif engine_id == "vibevoice":
            # Override voice_id from Orpheus selector with VibeVoice selector input
            voice_id = vibevoice_voice
            if not voice_id: 
                return None, "Please select a VibeVoice speaker."
        elif engine_id == "chatterbox":
            if reference_audio is None: 
                return None, "Please upload reference audio."
        
        # Handle zero-shot via voice selection (if voice_id is a path)
        if voice_id and os.path.exists(voice_id) and engine_id == "chatterbox":
            reference_audio = voice_id
            voice_id = "cloned"
            
        # Generate using unified service
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
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            f.write(result.audio_data)
            
        status = f"✅ Generated sample with {engine_id}"
        
        # Apply Enhanced Post-processing if requested
        if use_postprocessing:
            try:
                from audiobook.utils.audio_enhancer import audio_pipeline
                enhanced_path = os.path.join("static_files/voice_samples", f"enhanced_{filename}")
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
