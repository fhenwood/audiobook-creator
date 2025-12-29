"""
Generator utility functions for Audiobook Creator.

Contains text processing and validation utilities extracted from generator.py.
"""

import re
from word2number import w2n
from audiobook.tts.voice_mapping import get_narrator_voice_for_character, get_voice_for_character_score
from audiobook.tts.audio_utils import validate_file_path, get_ebook_metadata_with_cover


def sanitize_filename(text):
    """Remove or replace problematic characters from filenames."""
    text = text.replace("'", '').replace('"', '').replace('/', ' ').replace('.', ' ')
    text = text.replace(':', '').replace('?', '').replace('\\', '').replace('|', '')
    text = text.replace('*', '').replace('<', '').replace('>', '').replace('&', 'and')
    
    # Cleanup file name based on pattern ^[a-zA-Z0-9\-_./]+$
    regex = r"[^a-zA-Z0-9\-_./\s]"
    text = re.sub(regex, ' ', text, 0, re.MULTILINE) 
    
    # Normalize whitespace and trim
    text = ' '.join(text.split())
    
    return text


def is_only_punctuation(text):
    """
    Check if a line contains only punctuation marks without any actual words.
    
    Args:
        text (str): The text line to check
        
    Returns:
        bool: True if the line contains only punctuation, False otherwise
    """
    if not text or not text.strip():
        return True
    
    # Common punctuation marks
    punctuation_chars = set('.,!?;:"\'-—–…()[]{}«»""''*#@&%$^~`|\\/<>')
    
    # Remove all whitespace and check what remains
    text_no_space = ''.join(text.split())
    
    # If empty after removing spaces, consider it as punctuation only
    if not text_no_space:
        return True
    
    # Check if all remaining characters are punctuation
    text_without_punct = ''.join(c for c in text_no_space if c not in punctuation_chars)
    
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
    Finds the appropriate voice for a character based on their gender score.

    Args:
        character (str): The name of the character for whom the voice is being determined.
        character_gender_map (dict): A dictionary mapping character names to their gender scores.
        engine_name (str): The TTS engine name.
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
    
    Args:
        book_path (str): Path to the book file
        
    Returns:
        tuple: (is_valid, error_message, metadata)
    """
    try:
        if not validate_file_path(book_path):
            return False, f"Invalid or inaccessible book file: {book_path}.", None
        
        metadata = get_ebook_metadata_with_cover(book_path)
        
        if not metadata or len(metadata) == 0:
            return False, f"No metadata could be extracted from: {book_path}.", None
            
        if not validate_file_path("cover.jpg"):
            return False, f"Could not extract cover image from: {book_path}.", None
            
        return True, None, metadata
        
    except ValueError as e:
        return False, f"Book validation error: {str(e)}", None
    except RuntimeError as e:
        return False, f"Ebook processing error: {str(e)}", None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None
