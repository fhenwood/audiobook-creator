"""
Orpheus TTS Engine

Orpheus is a Llama-based TTS model optimized for natural, human-like speech
with emotion control and multiple voices.
"""

import os
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from audiobook.utils.gpu_resource_manager import gpu_manager

from audiobook.tts.engines.base import (
    TTSEngine, 
    VoiceInfo, 
    GenerationResult, 
    EngineConfig, 
    EngineCapability
)
from audiobook.tts.engines.registry import register_engine


# Orpheus voice definitions with metadata
ORPHEUS_VOICES = [
    VoiceInfo(
        id="zac",
        name="Zac",
        gender="male",
        description="Strong, confident narrator",
        tags=["narrator", "confident", "warm"]
    ),
    VoiceInfo(
        id="tara", 
        name="Tara",
        gender="female",
        description="Warm, expressive narrator voice",
        tags=["narrator", "warm", "expressive"]
    ),
    VoiceInfo(
        id="leah",
        name="Leah", 
        gender="female",
        description="Soft, gentle tone",
        tags=["soft", "gentle", "dialogue"]
    ),
    VoiceInfo(
        id="jess",
        name="Jess",
        gender="female", 
        description="Energetic, youthful voice",
        tags=["energetic", "youthful"]
    ),
    VoiceInfo(
        id="leo",
        name="Leo",
        gender="male",
        description="Deep, authoritative voice",
        tags=["deep", "authoritative"]
    ),
    VoiceInfo(
        id="dan",
        name="Dan",
        gender="male",
        description="Friendly, conversational tone",
        tags=["friendly", "conversational", "dialogue"]
    ),
    VoiceInfo(
        id="mia",
        name="Mia",
        gender="female",
        description="Clear, articulate voice",
        tags=["clear", "articulate"]
    ),
    VoiceInfo(
        id="zoe",
        name="Zoe",
        gender="female",
        description="Neutral, versatile voice",
        tags=["neutral", "versatile"]
    ),
]


@register_engine
class OrpheusEngine(TTSEngine):
    """
    Orpheus TTS Engine implementation.
    
    Uses the Orpheus TTS API (OpenAI-compatible) for speech generation.
    Supports emotion tags and multiple voices.
    """
    
    name = "orpheus"
    display_name = "Orpheus TTS"
    version = "1.0.0"
    
    capabilities = [
        EngineCapability.EMOTION_CONTROL,
        EngineCapability.MULTI_SPEAKER,
        EngineCapability.LONG_FORM,
        EngineCapability.PARALINGUISTIC,
    ]
    
    min_vram_gb = 4.0
    recommended_vram_gb = 6.0
    
    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        
        # Load configuration from environment or config
        # TTS_BASE_URL is the OpenAI-compatible TTS API (port 8880)
        # Note: ORPHEUS_API_URL is for llama.cpp token generation, not audio
        self.api_url = (
            config.api_url if config and config.api_url 
            else os.getenv("TTS_BASE_URL", "http://localhost:8880/v1")
        )
        self.api_key = (
            config.api_key if config and config.api_key
            else os.getenv("TTS_API_KEY", "not-needed")
        )
        self.model_name = (
            config.model_name if config and config.model_name
            else "orpheus"
        )
        self.timeout = config.timeout_seconds if config else 600
        
        self._client: Optional[AsyncOpenAI] = None
    
    async def initialize(self) -> bool:
        """Initialize the Orpheus API client."""
        try:
            self._client = AsyncOpenAI(
                base_url=self.api_url,
                api_key=self.api_key
            )
            
            # Start Orpheus LLM container
            success, msg = gpu_manager.acquire_orpheus()
            if not success:
                print(f"❌ Failed to acquire Orpheus resources: {msg}")
                return False
                
            self._initialized = True
            print(f"✅ {self.display_name} initialized at {self.api_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize {self.display_name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the engine and release GPU resources."""
        # Release GPU resources immediately to free VRAM for other engines
        gpu_manager.release_orpheus(immediate=True)
        
        self._client = None
        self._initialized = False
        print(f"🔓 {self.display_name} shutdown")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """Get all available Orpheus voices."""
        return ORPHEUS_VOICES.copy()
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float = 0.85,
        emotion: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate speech using Orpheus TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (e.g., "zac", "tara")
            speed: Speech speed multiplier (default 0.85)
            emotion: Optional emotion tag (e.g., "happy", "sad")
            **kwargs: Additional options
            
        Returns:
            GenerationResult with audio data
        """
        if not self._initialized or not self._client:
            await self.initialize()
        
        if not self.validate_voice(voice):
            raise ValueError(f"Invalid voice '{voice}' for {self.display_name}")
        
        # Format text with emotion if provided
        formatted_text = text
        if emotion:
            formatted_text = f"<{emotion}>{text}"
        
        try:
            audio_buffer = bytearray()
            
            async with self._client.audio.speech.with_streaming_response.create(
                model=self.model_name,
                voice=voice,
                response_format="wav",
                speed=speed,
                input=formatted_text,
                timeout=self.timeout
            ) as response:
                async for chunk in response.iter_bytes():
                    audio_buffer.extend(chunk)
            
            return GenerationResult(
                audio_data=bytes(audio_buffer),
                sample_rate=24000,
                voice_id=voice,
                metadata={
                    "engine": self.name,
                    "emotion": emotion,
                    "speed": speed
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Orpheus generation failed: {e}") from e
    
    async def health_check(self) -> bool:
        """Check if Orpheus API is accessible."""
        if not self._initialized:
            return False
        
        try:
            # Try generating a tiny audio sample
            result = await self.generate_speech("test", "zac")
            return len(result.audio_data) > 0
        except Exception:
            return False
    
    def get_default_voice(self) -> str:
        """Get default voice for Orpheus."""
        return "zac"
    
    def get_male_voices(self) -> List[VoiceInfo]:
        """Get all male voices."""
        return [v for v in ORPHEUS_VOICES if v.gender == "male"]
    
    def get_female_voices(self) -> List[VoiceInfo]:
        """Get all female voices."""
        return [v for v in ORPHEUS_VOICES if v.gender == "female"]

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        if self._client:
            await self._client.close()
        
        # Release resources
        gpu_manager.release_orpheus()

        self._initialized = False
        print(f"🔓 {self.display_name} shutdown")
