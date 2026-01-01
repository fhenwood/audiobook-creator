"""
Generator Package for Audiobook Creator.

This package contains modular components for audiobook audio generation.
The main generator.py is being progressively refactored into this package.
"""

# Re-export key functions from utils module
from audiobook.tts.generator.utils import (
    sanitize_filename,
    is_only_punctuation,
    split_and_annotate_text,
    check_if_chapter_heading,
    find_voice_for_gender_score,
    validate_book_for_m4b_generation,
)

# Export new modular components
from audiobook.tts.generator.text_loader import TextLoader
from audiobook.tts.generator.line_generator import LineGenerator
from audiobook.tts.generator.chapter_assembler import ChapterAssembler

# Import from the parent generator.py module using importlib
# This is needed because both generator/ (this package) and generator.py exist
import importlib.util
import os

_generator_module_path = os.path.join(os.path.dirname(__file__), "..", "generator.py")
_spec = importlib.util.spec_from_file_location("generator_module", os.path.abspath(_generator_module_path))
_generator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator_module)


process_audiobook_generation_v2 = _generator_module.process_audiobook_generation_v2

__all__ = [
    "sanitize_filename",
    "is_only_punctuation",
    "split_and_annotate_text",
    "check_if_chapter_heading",
    "find_voice_for_gender_score",
    "validate_book_for_m4b_generation",

    "process_audiobook_generation_v2",
    "TextLoader",
    "LineGenerator",
    "ChapterAssembler",
]
