"""
Audiobook TTS Module - Text-to-speech generation and voice management.
"""

from audiobook.tts.generator import (
    process_audiobook_generation,
    validate_book_for_m4b_generation,
    sanitize_filename,
)
from audiobook.tts.voice_mapping import (
    get_available_voices,
    get_voice_list,
    get_narrator_and_dialogue_voices,
)

__all__ = [
    "process_audiobook_generation",
    "validate_book_for_m4b_generation",
    "sanitize_filename",
    "get_available_voices",
    "get_voice_list",
    "get_narrator_and_dialogue_voices",
]
