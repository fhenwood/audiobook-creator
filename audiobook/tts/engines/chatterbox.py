"""
Chatterbox TTS Engine

Chatterbox is a high-performance TTS model from Resemble AI with 
zero-shot voice cloning capabilities.
"""

import os
from typing import List, Optional, Dict, Any, AsyncIterator
import aiohttp
import asyncio

from audiobook.tts.engines.base import (
    TTSEngine,
    VoiceInfo,
    GenerationResult,
    EngineConfig,
    EngineCapability
)
from audiobook.tts.engines.registry import register_engine


@register_engine
class ChatterboxEngine(TTSEngine):
    """
    Chatterbox TTS Engine implementation.
    
    Supports zero-shot voice cloning from a reference audio sample.
    Uses a local Chatterbox API server.
    """
    
    name = "chatterbox"
    display_name = "Chatterbox TTS"
    version = "1.0.0"
    
    capabilities = [
        EngineCapability.ZERO_SHOT_CLONING,
        EngineCapability.EMOTION_CONTROL,
        EngineCapability.MULTILINGUAL,
        EngineCapability.PARALINGUISTIC,
    ]
    
    min_vram_gb = 4.0
    recommended_vram_gb = 8.0
    
    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        
        # Load configuration from environment or config
        self.api_url = (
            config.api_url if config and config.api_url
            else os.getenv("CHATTERBOX_API_URL", "http://localhost:8881")
        )
        self.timeout = config.timeout_seconds if config else 600
        
        # Cache for voice conditionings
        self._voice_cache: Dict[str, Any] = {}
        self._reference_audio_path: Optional[str] = None
    
    async def initialize(self) -> bool:
        """Initialize the Chatterbox engine."""
        try:
            # Test connection to API
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self._initialized = True
                        print(f"✅ {self.display_name} initialized at {self.api_url}")
                        return True
        except Exception as e:
            print(f"⚠️ {self.display_name} health check failed (may not be running): {e}")
        
        # Even if health check fails, we can still mark as initialized
        # The actual error will occur at generation time
        self._initialized = True
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the engine and clear caches."""
        self._voice_cache.clear()
        self._reference_audio_path = None
        self._initialized = False
        print(f"🔓 {self.display_name} shutdown")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """
        Get available voices.
        
        For Chatterbox, voices are created dynamically from reference audio.
        Returns a placeholder for the cloned voice capability.
        """
        voices = [
            VoiceInfo(
                id="cloned",
                name="Cloned Voice",
                gender="neutral",
                description="Voice cloned from your reference audio",
                tags=["zero-shot", "cloned"]
            )
        ]
        
        # Add any cached voices
        for voice_id in self._voice_cache:
            if voice_id != "cloned":
                voices.append(VoiceInfo(
                    id=voice_id,
                    name=f"Cloned: {voice_id}",
                    gender="neutral",
                    description="Previously cloned voice",
                    tags=["cached", "cloned"]
                ))
        
        return voices
    
    def set_reference_audio(self, audio_path: str) -> None:
        """
        Set the reference audio for voice cloning.
        
        Args:
            audio_path: Path to the reference audio file
        """
        self._reference_audio_path = audio_path
        # Clear cache when reference changes
        self._voice_cache.clear()
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "cloned",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        reference_audio: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate speech using Chatterbox TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (typically "cloned" for Chatterbox)
            exaggeration: Emotion exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            reference_audio: Path to reference audio for cloning
            **kwargs: Additional options
            
        Returns:
            GenerationResult with audio data
        """
        if not self._initialized:
            await self.initialize()
        
        # Use provided reference or cached reference
        ref_audio = reference_audio or self._reference_audio_path
        if not ref_audio:
            raise ValueError(
                "Chatterbox requires a reference audio sample. "
                "Call set_reference_audio() first or provide reference_audio parameter."
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare multipart form data
                data = aiohttp.FormData()
                data.add_field('text', text)
                data.add_field('exaggeration', str(exaggeration))
                data.add_field('cfg_weight', str(cfg_weight))
                
                # Add reference audio file
                with open(ref_audio, 'rb') as f:
                    audio_content = f.read()
                data.add_field(
                    'reference_audio',
                    audio_content,
                    filename=os.path.basename(ref_audio),
                    content_type='audio/wav'
                )
                
                async with session.post(
                    f"{self.api_url}/generate",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Chatterbox API error: {error_text}")
                    
                    audio_data = await response.read()
                    
                    return GenerationResult(
                        audio_data=audio_data,
                        sample_rate=24000,
                        voice_id=voice,
                        metadata={
                            "engine": self.name,
                            "exaggeration": exaggeration,
                            "cfg_weight": cfg_weight,
                            "reference_audio": os.path.basename(ref_audio)
                        }
                    )
        
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Chatterbox connection error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Chatterbox generation failed: {e}") from e
    
    async def clone_voice(
        self,
        audio_sample: bytes,
        voice_name: str,
        **kwargs
    ) -> VoiceInfo:
        """
        Clone a voice from an audio sample.
        
        Args:
            audio_sample: Raw audio bytes
            voice_name: Name for the cloned voice
            **kwargs: Additional options
            
        Returns:
            VoiceInfo for the cloned voice
        """
        import tempfile
        
        # Save audio sample to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_sample)
            temp_path = f.name
        
        # Cache the voice
        self._voice_cache[voice_name] = {
            "audio_path": temp_path,
            "audio_bytes": audio_sample
        }
        
        return VoiceInfo(
            id=voice_name,
            name=f"Cloned: {voice_name}",
            gender="neutral",
            description=f"Voice cloned from audio sample",
            tags=["cloned", "zero-shot"]
        )
    
    async def health_check(self) -> bool:
        """Check if Chatterbox API is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def get_default_voice(self) -> str:
        """Get default voice (cloned)."""
        return "cloned"
