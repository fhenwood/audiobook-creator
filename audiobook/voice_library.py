"""
Voice Library - Persistence layer for cloned voices.

Stores voice configurations including:
- Voice name and description
- Reference audio path
- Reference transcript
- Qwen-generated voice description
- Optional SNAC tokens (pre-encoded for faster generation)
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)

VOICE_LIBRARY_PATH = "voice_library.json"


@dataclass
class ClonedVoice:
    """A cloned voice configuration."""
    id: str
    name: str
    reference_audio_path: str
    reference_transcript: str
    voice_description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    snac_tokens: Optional[List[int]] = None  # Pre-encoded for faster generation
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClonedVoice":
        return cls(**data)


class VoiceLibrary:
    """
    Persistent storage for cloned voices.
    
    Voices are stored in a JSON file with optional pre-encoded SNAC tokens.
    """
    
    def __init__(self, library_path: str = VOICE_LIBRARY_PATH):
        self.library_path = library_path
        self._voices: Dict[str, ClonedVoice] = {}
        self._load()
    
    def _load(self) -> None:
        """Load voices from disk."""
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for voice_data in data.get("voices", []):
                        voice = ClonedVoice.from_dict(voice_data)
                        self._voices[voice.id] = voice
                logger.info(f"📚 Loaded {len(self._voices)} voices from library")
            except Exception as e:
                logger.error(f"Failed to load voice library: {e}")
                self._voices = {}
    
    def _save(self) -> None:
        """Save voices to disk."""
        try:
            data = {
                "version": "1.0",
                "voices": [v.to_dict() for v in self._voices.values()]
            }
            with open(self.library_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(self._voices)} voices to library")
        except Exception as e:
            logger.error(f"Failed to save voice library: {e}")
    
    def add_voice(self, voice: ClonedVoice) -> None:
        """Add or update a voice in the library."""
        self._voices[voice.id] = voice
        self._save()
        logger.info(f"➕ Added voice '{voice.name}' to library")
    
    def get_voice(self, voice_id: str) -> Optional[ClonedVoice]:
        """Get a voice by ID."""
        return self._voices.get(voice_id)
    
    def list_voices(self) -> List[ClonedVoice]:
        """List all voices in the library."""
        return list(self._voices.values())
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice from the library."""
        if voice_id in self._voices:
            del self._voices[voice_id]
            self._save()
            logger.info(f"🗑️ Deleted voice '{voice_id}' from library")
            return True
        return False
    
    def get_choices(self) -> List[tuple]:
        """Get voice choices for Gradio dropdown."""
        choices = []
        for v in self._voices.values():
            label = f"🎤 {v.name}"
            if v.tags:
                label += f" [{', '.join(v.tags[:2])}]"
            choices.append((label, v.id))
        return choices


# Global instance
voice_library = VoiceLibrary()
