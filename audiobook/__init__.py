"""
Audiobook Creator Package
Copyright (C) 2025

A Python library for creating audiobooks from ebooks using TTS engines.
"""

from audiobook.core.text_extraction import (
    extract_text_from_book_using_calibre,
    process_book_and_extract_text,
    save_book,
)
from audiobook.core.emotion_tags import process_emotion_tags
from audiobook.core.character_identification import (
    identify_characters_and_output_book_to_jsonl,
    process_book_and_identify_characters,
)
from audiobook.tts.generator import (
    process_audiobook_generation,
    validate_book_for_m4b_generation,
    sanitize_filename,
)
from audiobook.tts.voice_mapping import (
    get_available_voices,
    get_voice_list,
)

__version__ = "1.0.0"
__all__ = [
    "extract_text_from_book_using_calibre",
    "process_book_and_extract_text",
    "save_book",
    "process_emotion_tags",
    "identify_characters_and_output_book_to_jsonl",
    "process_book_and_identify_characters",
    "process_audiobook_generation",
    "validate_book_for_m4b_generation",
    "sanitize_filename",
    "get_available_voices",
    "get_voice_list",
]
