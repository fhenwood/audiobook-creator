"""
Voice and TTS utility functions for Audiobook Creator.

Extracted from app.py to keep the main application file lean.
"""

import os
from audiobook.tts.voice_mapping import get_available_voices, get_voice_list
from audiobook.tts.generator import sanitize_filename
from audiobook.core.voice_manager import voice_manager
from audiobook.models.manager import model_manager, ModelType


def extract_title_from_filename(book_file):
    """Extract book title from uploaded filename without extension"""
    if book_file is None:
        return ""
    filename = os.path.basename(book_file.name)
    title = os.path.splitext(filename)[0]
    return sanitize_filename(title)


def get_vibevoice_choices():
    """Get VibeVoice choices from Voice Library only (no generic speakers)."""
    # Only return custom voices from Voice Library (no generic speaker_0-3)
    custom = [(f"🎤 {v.name}", v.path) for v in voice_manager.list_voices()]
    if not custom:
        # Fallback if no voices uploaded yet
        return [("No voices - add some in Voice Library", "")]
    return custom


def get_voice_choices():
    """Get voice choices for dropdown with descriptions."""
    # Standard voices
    voices = get_available_voices()
    choices = [(f"{name} - {desc}", name) for name, desc in voices.items()]
    
    # Custom voices
    custom_voices = voice_manager.list_voices()
    for v in custom_voices:
        choices.append((f"🎤 {v.name} (Custom)", v.path))
        
    return choices


def get_installed_tts_engines():
    """Get list of installed TTS engines."""
    engines = []
    
    # Orpheus is the core engine, assumed available via container services
    engines.append("Orpheus")
    
    # Chatterbox requires XTTS v2
    if model_manager.get_model_path("xtts-v2"):
        engines.append("Chatterbox")
        
    # VibeVoice requires vibevoice-7b
    if model_manager.get_model_path("vibevoice-7b"):
        engines.append("VibeVoice")
        
    if not engines:
        engines.append("(No Engines Installed)")
        
    return engines


def check_llm_availability():
    """Check if any LLM is installed for emotion tagging."""
    models = model_manager.list_models()
    for m in models:
        if m.definition.type == ModelType.LLM and m.installed:
            return True
    return False
