"""
Generator Package for Audiobook Creator.

This package contains modular components for audiobook audio generation.
The main generator.py is being progressively refactored into this package.
"""

# Re-export key functions from parent generator.py for backward compatibility
# These will be moved into this package progressively

from audiobook.tts.generator.utils import (
    sanitize_filename,
    is_only_punctuation,
    split_and_annotate_text,
    check_if_chapter_heading,
    find_voice_for_gender_score,
    validate_book_for_m4b_generation,
)

__all__ = [
    "sanitize_filename",
    "is_only_punctuation",
    "split_and_annotate_text",
    "check_if_chapter_heading",
    "find_voice_for_gender_score",
    "validate_book_for_m4b_generation",
]
