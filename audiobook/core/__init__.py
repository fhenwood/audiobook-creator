"""
Audiobook Core Module - Text extraction, processing, and character identification.
"""

from audiobook.core.text_extraction import (
    extract_text_from_book_using_calibre,
    process_book_and_extract_text,
    save_book,
    validate_book_path,
)
from audiobook.core.emotion_tags import process_emotion_tags
from audiobook.core.character_identification import (
    identify_characters_and_output_book_to_jsonl,
    process_book_and_identify_characters,
)

__all__ = [
    "extract_text_from_book_using_calibre",
    "process_book_and_extract_text",
    "save_book",
    "validate_book_path",
    "process_emotion_tags",
    "identify_characters_and_output_book_to_jsonl",
    "process_book_and_identify_characters",
]
