"""
TTS Engine Abstraction Layer

This module provides the base interface for all TTS engines in the audiobook creator.
New TTS engines can be added by implementing the TTSEngine abstract base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, AsyncIterator
import asyncio


class EngineCapability(Enum):
    """Capabilities that a TTS engine may support."""
    ZERO_SHOT_CLONING = auto()      # Clone voice from audio sample
    MULTI_SPEAKER = auto()           # Multiple distinct speakers
    EMOTION_CONTROL = auto()         # Control emotional expression
    LONG_FORM = auto()               # Generate long audio (>5 min)
    STREAMING = auto()               # Real-time audio streaming
    MULTILINGUAL = auto()            # Multiple languages
    PARALINGUISTIC = auto()          # Sounds like laughs, sighs, etc.


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    id: str                          # Unique identifier for the engine
    name: str                        # Display name
    gender: str                      # "male", "female", "neutral"
    description: str = ""            # Optional description
    language: str = "en"             # Language code
    sample_url: Optional[str] = None # URL to voice sample
    tags: List[str] = field(default_factory=list)  # e.g., ["warm", "professional"]
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class GenerationResult:
    """Result of a TTS generation request."""
    audio_data: bytes                # Raw audio bytes (WAV format preferred)
    sample_rate: int = 24000         # Audio sample rate
    duration_seconds: float = 0.0    # Duration of generated audio
    voice_id: str = ""               # Voice used for generation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class EngineConfig:
    """Configuration for a TTS engine."""
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    timeout_seconds: int = 600
    max_retries: int = 3
    gpu_enabled: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.
    
    All TTS engines must implement this interface to be compatible
    with the audiobook generation system.
    
    Example usage:
        class MyTTSEngine(TTSEngine):
            name = "mytts"
            display_name = "My TTS Engine"
            
            async def generate_speech(self, text, voice, **kwargs):
                # Implementation here
                return GenerationResult(audio_data=...)
    """
    
    # Engine identification (must be overridden)
    name: str = "base"                    # Unique identifier (lowercase, no spaces)
    display_name: str = "Base Engine"     # Human-readable name
    version: str = "1.0.0"
    
    # Capabilities (override in subclass)
    capabilities: List[EngineCapability] = []
    
    # Resource requirements
    min_vram_gb: float = 0.0              # Minimum VRAM required
    recommended_vram_gb: float = 0.0       # Recommended VRAM
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the engine with optional configuration."""
        self.config = config or EngineConfig()
        self._initialized = False
    
    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    async def generate_speech(
        self, 
        text: str, 
        voice: str,
        **kwargs
    ) -> GenerationResult:
        """
        Generate speech audio from text.
        
        Args:
            text: The text to convert to speech
            voice: Voice identifier to use
            **kwargs: Engine-specific options (emotion, speed, etc.)
            
        Returns:
            GenerationResult containing the audio data
            
        Raises:
            ValueError: If voice is invalid
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[VoiceInfo]:
        """
        Get list of available voices for this engine.
        
        Returns:
            List of VoiceInfo objects describing available voices
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the engine (load models, connect to API, etc.)
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the engine and release resources.
        """
        pass
    
    # =========================================================================
    # Optional Methods (Can be overridden for enhanced functionality)
    # =========================================================================
    
    def validate_voice(self, voice_id: str) -> bool:
        """
        Check if a voice ID is valid for this engine.
        
        Args:
            voice_id: The voice identifier to validate
            
        Returns:
            True if voice is valid, False otherwise
        """
        valid_ids = [v.id for v in self.get_available_voices()]
        return voice_id in valid_ids
    
    def get_voice_by_id(self, voice_id: str) -> Optional[VoiceInfo]:
        """Get a specific voice by its ID."""
        for voice in self.get_available_voices():
            if voice.id == voice_id:
                return voice
        return None
    
    def get_voices_by_gender(self, gender: str) -> List[VoiceInfo]:
        """Get all voices of a specific gender."""
        return [v for v in self.get_available_voices() if v.gender.lower() == gender.lower()]
    
    async def generate_speech_streaming(
        self, 
        text: str, 
        voice: str,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Generate speech with streaming output.
        
        Override this for engines that support real-time streaming.
        Default implementation buffers and yields the full result.
        
        Args:
            text: The text to convert to speech
            voice: Voice identifier to use
            **kwargs: Engine-specific options
            
        Yields:
            Audio data chunks as they become available
        """
        result = await self.generate_speech(text, voice, **kwargs)
        yield result.audio_data
    
    def supports(self, capability: EngineCapability) -> bool:
        """Check if this engine supports a specific capability."""
        return capability in self.capabilities
    
    async def clone_voice(
        self, 
        audio_sample: bytes, 
        voice_name: str,
        **kwargs
    ) -> VoiceInfo:
        """
        Clone a voice from an audio sample (zero-shot).
        
        Only available if engine supports ZERO_SHOT_CLONING capability.
        
        Args:
            audio_sample: Audio bytes of the voice to clone
            voice_name: Name to give the cloned voice
            **kwargs: Engine-specific cloning options
            
        Returns:
            VoiceInfo for the newly cloned voice
            
        Raises:
            NotImplementedError: If engine doesn't support voice cloning
        """
        if not self.supports(EngineCapability.ZERO_SHOT_CLONING):
            raise NotImplementedError(f"{self.display_name} does not support voice cloning")
        raise NotImplementedError("Subclass must implement clone_voice()")
    
    async def health_check(self) -> bool:
        """
        Check if the engine is healthy and ready to generate speech.
        
        Returns:
            True if engine is ready, False otherwise
        """
        return self._initialized
    
    def get_default_voice(self) -> Optional[str]:
        """Get the default voice ID for this engine."""
        voices = self.get_available_voices()
        return voices[0].id if voices else None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about this engine."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "capabilities": [c.name for c in self.capabilities],
            "min_vram_gb": self.min_vram_gb,
            "recommended_vram_gb": self.recommended_vram_gb,
            "voice_count": len(self.get_available_voices()),
            "initialized": self._initialized
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, initialized={self._initialized})>"
