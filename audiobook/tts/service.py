"""
Unified TTS Generation Service

This module provides the central interface for TTS generation across all engines.
It abstracts away the specific engine implementation and provides a clean API
for the rest of the application.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from audiobook.tts.engines import (
    engine_registry,
    get_engine,
    list_engines,
    TTSEngine,
    VoiceInfo,
    GenerationResult,
    EngineCapability
)


@dataclass
class TTSRequest:
    """Request for TTS generation."""
    text: str
    engine_name: str = "orpheus"
    voice: Optional[str] = None  # None = use default
    speed: float = 1.0
    emotion: Optional[str] = None
    reference_audio: Optional[str] = None  # For zero-shot cloning
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class TTSService:
    """
    Unified TTS service that manages engine selection and generation.
    
    This is the single entry point for all TTS operations.
    Handles engine lifecycle, GPU management, and provides a clean API.
    
    Usage:
        service = TTSService()
        
        # Generate with default engine (Orpheus)
        result = await service.generate("Hello world")
        
        # Generate with specific engine
        result = await service.generate("Hello", engine="chatterbox", reference_audio="voice.wav")
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton to ensure one service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_engine: Optional[str] = None
        self._engine_instances: Dict[str, TTSEngine] = {}
    
    async def generate(
        self,
        text: str,
        engine: str = "orpheus",
        voice: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate speech from text using the specified engine.
        
        Args:
            text: Text to convert to speech
            engine: Engine name (orpheus, chatterbox, vibevoice)
            voice: Voice ID (uses default if None)
            **kwargs: Engine-specific parameters
            
        Returns:
            GenerationResult with audio data
            
        Raises:
            ValueError: If engine not found
            RuntimeError: If generation fails
        """
        tts_engine = await self._get_or_load_engine(engine)
        
        # Use default voice if not specified
        if voice is None:
            voice = tts_engine.get_default_voice()
        
        # Generate audio
        result = await tts_engine.generate_speech(text, voice, **kwargs)
        return result
    
    async def generate_batch(
        self,
        texts: List[str],
        engine: str = "orpheus",
        voice: Optional[str] = None,
        concurrency: int = 4,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate speech for multiple texts with concurrency control.
        
        Args:
            texts: List of texts to convert
            engine: Engine name
            voice: Voice ID
            concurrency: Max parallel generations
            **kwargs: Engine-specific parameters
            
        Returns:
            List of GenerationResults (in same order as input)
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def generate_one(text: str) -> GenerationResult:
            async with semaphore:
                return await self.generate(text, engine, voice, **kwargs)
        
        results = await asyncio.gather(
            *[generate_one(text) for text in texts],
            return_exceptions=True
        )
        
        # Handle exceptions
        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ Generation failed for text {i}: {result}")
                output.append(None)
            else:
                output.append(result)
        
        return output
    
    async def _get_or_load_engine(self, engine_name: str) -> TTSEngine:
        """Get engine instance, loading if necessary and unloading others."""
        
        # If already current and initialized, return it
        if self._current_engine == engine_name:
            engine = self._engine_instances.get(engine_name)
            if engine and engine._initialized:
                return engine
        
        # If switching engines, shutdown the previous one to free VRAM
        if self._current_engine and self._current_engine != engine_name:
            print(f"🔄 Switching engine: Unloading {self._current_engine}...")
            prev_engine = self._engine_instances.get(self._current_engine)
            if prev_engine:
                await prev_engine.shutdown()
            self._current_engine = None
        
        # Get engine from registry
        engine = get_engine(engine_name)
        if engine is None:
            available = list_engines()
            raise ValueError(
                f"Engine '{engine_name}' not found. "
                f"Available: {available}"
            )
        
        # Initialize engine
        if not engine._initialized:
            print(f"🔄 Initializing {engine.display_name}...")
            success = await engine.initialize()
            if not success:
                # If intialization failed (e.g. no GPU), don't set as current
                raise RuntimeError(
                    f"Failed to initialize {engine.display_name}. "
                    "Check logs for details (missing GPU?)"
                )
            print(f"✅ {engine.display_name} ready")
        
        self._engine_instances[engine_name] = engine
        self._current_engine = engine_name
        return engine
    
    def get_available_engines(self) -> List[Dict]:
        """Get info about all available engines."""
        return engine_registry.get_engine_info_all()
    
    def get_voices_for_engine(self, engine_name: str) -> List[VoiceInfo]:
        """Get available voices for a specific engine."""
        engine = get_engine(engine_name)
        if engine is None:
            return []
        return engine.get_available_voices()
    
    def get_all_voices(self) -> Dict[str, List[VoiceInfo]]:
        """Get all voices from all engines."""
        return engine_registry.get_all_voices()
    
    async def health_check(self, engine_name: str) -> bool:
        """Check if an engine is healthy."""
        try:
            engine = await self._get_or_load_engine(engine_name)
            return await engine.health_check()
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all engines and release resources."""
        await engine_registry.shutdown_all()
        self._engine_instances.clear()
        self._current_engine = None
        print("🔓 TTS Service shutdown complete")


# Global service instance
tts_service = TTSService()


# Convenience functions for backward compatibility
async def generate_tts(
    text: str,
    engine: str = "orpheus",
    voice: Optional[str] = None,
    **kwargs
) -> bytes:
    """
    Generate TTS audio (convenience function).
    
    Returns raw audio bytes (WAV format).
    """
    result = await tts_service.generate(text, engine, voice, **kwargs)
    return result.audio_data


def get_available_engines() -> List[str]:
    """Get list of available engine names."""
    return list_engines()


def get_voices(engine: str = "orpheus") -> List[VoiceInfo]:
    """Get voices for an engine."""
    return tts_service.get_voices_for_engine(engine)
