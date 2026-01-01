"""
Audiobook Creator

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
from audiobook.utils.file_utils import read_json

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


def get_narrator_voice_for_character(engine_name: str, narrator_gender: str) -> str:
    """
    Get the narrator voice for a character based on engine and gender.
    
    Args:
        engine_name (str): TTS engine name (e.g., "orpheus")
        narrator_gender (str): Gender of narrator ("male" or "female")
    
    Returns:
        str: Voice identifier for narrator
    """
    narrator_voice, _ = get_narrator_and_dialogue_voices(engine_name, narrator_gender)
    return narrator_voice


def get_voice_for_character_score(engine_name: str, narrator_gender: str, gender_score: int) -> str:
    """
    Get voice for a character based on their gender score.
    
    Gender score ranges from 0-10:
    - 0-3: Strong female voice
    - 4-6: Neutral/narrator voice  
    - 7-10: Strong male voice
    
    Args:
        engine_name (str): TTS engine name (e.g., "orpheus")
        narrator_gender (str): Gender of narrator ("male" or "female")
        gender_score (int): Character's gender score (0-10)
    
    Returns:
        str: Voice identifier matching the character's gender profile
    """
    voice_mappings = load_voice_mappings()
    engine_voices = voice_mappings.get(engine_name, voice_mappings.get("orpheus", {}))
    
    # Get score-to-voice mapping or use defaults
    if narrator_gender == "male":
        score_map = engine_voices.get("male_score_map", {
            "0": "leah", "1": "leah", "2": "jess", "3": "mia",
            "4": "zoe", "5": "zac", "6": "zac",
            "7": "dan", "8": "dan", "9": "leo", "10": "leo"
        })
    else:
        score_map = engine_voices.get("female_score_map", {
            "0": "leah", "1": "leah", "2": "jess", "3": "mia",
            "4": "tara", "5": "tara", "6": "zoe",
            "7": "dan", "8": "dan", "9": "leo", "10": "leo"
        })
    
    # Clamp score to valid range
    score = max(0, min(10, gender_score))
    return score_map.get(str(score), "zac" if narrator_gender == "male" else "tara")