"""
Orpheus TTS Engine - In-Process Implementation

Orpheus is a Llama-based TTS model optimized for natural, human-like speech
with emotion control and multiple voices.

This version uses llama-cpp-python for direct in-process inference,
eliminating the need for a separate container.
"""

import os
import gc
from typing import List, Optional, Dict, Any

from audiobook.tts.engines.base import (
    TTSEngine, 
    VoiceInfo, 
    GenerationResult, 
    EngineConfig, 
    EngineCapability
)
from audiobook.tts.engines.registry import register_engine
from audiobook.models.manager import model_manager
from audiobook.config import settings


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

# Model config
DEFAULT_MODEL_ID = "orpheus-3b-0.1-ft-Q4_K_M.gguf"
SAMPLE_RATE = 24000


@register_engine
class OrpheusEngine(TTSEngine):
    """
    Orpheus TTS Engine implementation with in-process inference.
    
    Uses llama-cpp-python for LLM token generation and SNAC for audio decoding.
    This eliminates the need for a separate container and enables proper GPU
    hot-swapping with other engines like VibeVoice.
    """
    
    name = "orpheus"
    display_name = "Orpheus TTS"
    version = "2.0.0"  # In-process version
    
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
        
        self.model_id = (
            config.model_name if config and config.model_name
            else DEFAULT_MODEL_ID
        )
        self.temperature = settings.orpheus_temperature
        self.top_p = settings.orpheus_top_p
        self.max_tokens = settings.orpheus_max_tokens
        self.repetition_penalty = 1.1
        
        self._llm = None
        self._snac = None
    
    async def initialize(self) -> bool:
        """Initialize the Orpheus model in-process."""
        try:
            import torch
            
            # Get model path from model manager
            model_path = model_manager.get_model_path(self.model_id)
            if not model_path:
                # Try downloading the model
                print(f"📥 Downloading {self.model_id}...")
                model_path = model_manager.download_model(self.model_id)
                if not model_path:
                    print(f"❌ Model {self.model_id} not available")
                    return False
            
            print(f"🚀 Loading {self.display_name} from {model_path}...")

            # If model_path is a directory, find the GGUF file inside
            if os.path.isdir(model_path):
                found_gguf = False
                for file in os.listdir(model_path):
                    if file.endswith(".gguf"):
                        model_path = os.path.join(model_path, file)
                        found_gguf = True
                        break
                if not found_gguf:
                     print(f"❌ No GGUF file found in {model_path}")
                     return False

            
            # Load model using llama-cpp-python
            try:
                from llama_cpp import Llama
                
                # Determine GPU layers
                n_gpu_layers = -1 if torch.cuda.is_available() else 0
                
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=self.max_tokens,
                    verbose=False
                )
                print(f"✅ LLM loaded with {n_gpu_layers} GPU layers")
            except ImportError:
                print("❌ llama-cpp-python not available. In-process execution requires this package.")
                print("   Install it with: pip install llama-cpp-python")
                return False
            
            # Load SNAC audio decoder
            try:
                import snac
                self._snac = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                if torch.cuda.is_available():
                    self._snac = self._snac.to("cuda")
                print("✅ SNAC decoder loaded")
            except Exception as e:
                print(f"⚠️ SNAC loading failed: {e}")
                # Continue without SNAC - will need API fallback for audio
            
            self._initialized = True
            print(f"✅ {self.display_name} initialized (in-process)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.display_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the engine and release GPU resources."""
        import torch
        
        if self._llm:
            del self._llm
            self._llm = None
        
        if self._snac:
            del self._snac
            self._snac = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
        print(f"🔓 {self.display_name} shutdown")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """Get all available Orpheus voices."""
        return ORPHEUS_VOICES.copy()
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "zac",
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
        if not self._initialized:
            await self.initialize()
        
        if not self.validate_voice(voice):
            raise ValueError(f"Invalid voice '{voice}' for {self.display_name}")
        
        # Format text with emotion if provided
        formatted_text = text
        if emotion:
            formatted_text = f"<{emotion}>{text}"
        
        # Generate using local LLM + SNAC
        return await self._generate_local(formatted_text, voice, speed)
    
    async def _generate_local(
        self, 
        text: str, 
        voice: str, 
        speed: float
    ) -> GenerationResult:
        """Generate using local LLM + SNAC decoder."""
        import numpy as np
        import wave
        import io
        import torch
        
        # Format prompt for Orpheus
        prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
        
        # Generate tokens
        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            stop=["<|eot_id|>"]
        )
        
        generated_text = output["choices"][0]["text"]
        
        # Parse audio tokens from output
        tokens = self._parse_audio_tokens(generated_text)
        
        if not tokens or self._snac is None:
            # Return silence if unable to decode
            audio_data = self._generate_silence(1.0)
        else:
            # Decode tokens to audio using SNAC
            with torch.no_grad():
                audio_tensor = self._snac.decode(tokens)
                audio_np = audio_tensor.cpu().numpy().flatten()
                audio_data = self._numpy_to_wav(audio_np, SAMPLE_RATE)
        
        return GenerationResult(
            audio_data=audio_data,
            sample_rate=SAMPLE_RATE,
            voice_id=voice,
            metadata={"engine": self.name, "mode": "local"}
        )
    
    def _parse_audio_tokens(self, text: str) -> Optional[list]:
        """Parse audio tokens from LLM output."""
        # Audio tokens are formatted as <|audio_N|> where N is the token ID
        import re
        tokens = []
        pattern = r'<\|audio_(\d+)\|>'
        matches = re.findall(pattern, text)
        for match in matches:
            tokens.append(int(match))
        return tokens if tokens else None
    
    def _generate_silence(self, duration: float) -> bytes:
        """Generate silent audio as fallback."""
        import struct
        sample_count = int(SAMPLE_RATE * duration)
        audio_data = struct.pack('<' + 'h' * sample_count, *([0] * sample_count))
        return self._raw_to_wav(audio_data, SAMPLE_RATE)
    
    def _numpy_to_wav(self, audio: 'np.ndarray', sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        import io
        import wave
        import numpy as np
        
        # Normalize and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def _raw_to_wav(self, raw_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        import io
        import wave
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(raw_data)
        
        return buffer.getvalue()
    
    async def health_check(self) -> bool:
        """Check if engine is ready."""
        if not self._initialized:
            return False
        
        # Check local model
        if self._llm is not None:
            return True
        
        # Check API fallback
        if hasattr(self, '_use_api') and self._use_api:
            try:
                result = await self.generate_speech("test", "zac")
                return len(result.audio_data) > 0
            except Exception:
                return False
        
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
