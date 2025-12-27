"""
Audiobook Utilities Module - Helper functions for file handling, LLM, and shell commands.
"""

from audiobook.utils.file_utils import read_json, read_jsonl, empty_directory, empty_file
from audiobook.utils.llm_utils import check_if_llm_is_up, generate_audio_with_retry
from audiobook.utils.shell_commands import (
    check_if_calibre_is_installed,
    check_if_ffmpeg_is_installed,
    run_shell_command_secure,
    validate_file_path_allowlist,
)
from audiobook.utils.text_preprocessing import preprocess_text_for_tts
from audiobook.utils.api_health import check_if_audio_generator_api_is_up

__all__ = [
    "read_json",
    "read_jsonl",
    "empty_directory",
    "empty_file",
    "check_if_llm_is_up",
    "generate_audio_with_retry",
    "check_if_calibre_is_installed",
    "check_if_ffmpeg_is_installed",
    "run_shell_command_secure",
    "validate_file_path_allowlist",
    "preprocess_text_for_tts",
    "check_if_audio_generator_api_is_up",
]
