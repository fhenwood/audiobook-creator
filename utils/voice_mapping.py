"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

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

import os
from utils.file_utils import read_json

def load_voice_mappings():
    """Load the voice mappings from the JSON file."""
    return read_json("static_files/voice_map.json")

def get_available_voices():
    """
    Get list of available Orpheus TTS voices with descriptions.
    
    Returns:
        dict: Dictionary with voice names as keys and descriptions as values
    """
    voice_mappings = load_voice_mappings()
    orpheus_config = voice_mappings.get("orpheus", {})
    return orpheus_config.get("voice_descriptions", {})

def get_voice_list():
    """
    Get simple list of available voice names.
    
    Returns:
        list: List of voice name strings
    """
    voice_mappings = load_voice_mappings()
    orpheus_config = voice_mappings.get("orpheus", {})
    return orpheus_config.get("voices", ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"])

def get_narrator_and_dialogue_voices(engine_name: str = "orpheus", narrator_gender: str = "female"):
    """
    Get narrator and dialogue voices for single-voice mode.
    
    Args:
        engine_name (str): TTS engine name (always "orpheus")
        narrator_gender (str): Gender of narrator ("male" or "female")
    
    Returns:
        tuple: (narrator_voice, dialogue_voice)
    """
    voice_mappings = load_voice_mappings()
    
    # Always use orpheus
    engine_voices = voice_mappings.get("orpheus", {})
    
    if narrator_gender == "male":
        narrator_voice = engine_voices.get("male_narrator", "zac")
        dialogue_voice = engine_voices.get("male_dialogue", "dan")
    else:  # female
        narrator_voice = engine_voices.get("female_narrator", "tara")
        dialogue_voice = engine_voices.get("female_dialogue", "leah")
    
    return narrator_voice, dialogue_voice

def get_narrator_voice(narrator_gender: str = "female"):
    """
    Get just the narrator voice based on gender preference.
    
    Args:
        narrator_gender (str): Gender of narrator ("male" or "female")
    
    Returns:
        str: Narrator voice identifier
    """
    narrator_voice, _ = get_narrator_and_dialogue_voices("orpheus", narrator_gender)
    return narrator_voice

def get_dialogue_voice(narrator_gender: str = "female"):
    """
    Get just the dialogue voice based on gender preference.
    
    Args:
        narrator_gender (str): Gender of narrator ("male" or "female")
    
    Returns:
        str: Dialogue voice identifier
    """
    _, dialogue_voice = get_narrator_and_dialogue_voices("orpheus", narrator_gender)
    return dialogue_voice 