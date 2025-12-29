"""
VibeVoice TTS Engine

VibeVoice is Microsoft's open-source TTS model for expressive, 
long-form, multi-speaker conversational audio generation.
Supports up to 4 distinct speakers and 90-minute generation.
"""

import os
import asyncio
from typing import List, Optional, Dict, Any

from audiobook.tts.engines.base import (
    TTSEngine,
    VoiceInfo,
    GenerationResult,
    EngineConfig,
    EngineCapability
)
from audiobook.tts.engines.registry import register_engine
from audiobook.utils.gpu_resource_manager import gpu_manager
from audiobook.models.manager import model_manager


# VibeVoice default speakers
VIBEVOICE_SPEAKERS = [
    VoiceInfo(
        id="speaker_0",
        name="Speaker A",
        gender="neutral",
        description="Primary speaker voice",
        tags=["conversational", "main"]
    ),
    VoiceInfo(
        id="speaker_1",
        name="Speaker B", 
        gender="neutral",
        description="Secondary speaker voice",
        tags=["conversational", "dialogue"]
    ),
    VoiceInfo(
        id="speaker_2",
        name="Speaker C",
        gender="neutral",
        description="Tertiary speaker voice",
        tags=["conversational"]
    ),
    VoiceInfo(
        id="speaker_3",
        name="Speaker D",
        gender="neutral",
        description="Quaternary speaker voice",
        tags=["conversational"]
    ),
]


@register_engine
class VibeVoiceEngine(TTSEngine):
    """
    VibeVoice TTS Engine implementation.
    
    Microsoft's expressive, long-form, multi-speaker TTS model.
    Requires significant VRAM (19-24GB for 7B model).
    
    Features:
    - Multi-speaker support (up to 4 voices)
    - Long-form generation (up to 90 minutes)
    - Zero-shot voice cloning
    - Podcast/conversation generation
    """
    
    name = "vibevoice"
    display_name = "VibeVoice 7B"
    version = "1.0.0"
    
    capabilities = [
        EngineCapability.ZERO_SHOT_CLONING,
        EngineCapability.MULTI_SPEAKER,
        EngineCapability.LONG_FORM,
        EngineCapability.EMOTION_CONTROL,
    ]
    
    min_vram_gb = 19.0
    recommended_vram_gb = 24.0
    
    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        
        # Model settings
        self.model_size = os.getenv("VIBEVOICE_MODEL_SIZE", "7b")  # 1.5b or 7b
        self._model = None
        
        # Voice cloning cache
        self._cloned_voices: Dict[str, bytes] = {}
    
    async def initialize(self) -> bool:
        """Initialize VibeVoice engine."""
        try:
            import torch
            
            # CPU/GPU Check
            if not torch.cuda.is_available():
                print(f"⚠️ {self.display_name} requires CUDA GPU. Skipping initialization.")
                return False
                
            print(f"🔄 Loading {self.display_name} model ({self.model_size})...")
            
            # Acquire GPU resources (unloads other models)
            success, msg = gpu_manager.acquire_vibevoice()
            if not success:
                print(f"❌ Failed to acquire GPU for {self.display_name}: {msg}")
                return False

            # Lazy import to avoid loading heavy deps if not used
            try:
                # Use inference-specific model class that supports .generate()
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference as VibeVoiceForConditionalGeneration
                VIBEVOICE_AVAILABLE = True
            except ImportError:
                try:
                    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
                    VIBEVOICE_AVAILABLE = True
                except ImportError:
                    VIBEVOICE_AVAILABLE = False          
            
            if not VIBEVOICE_AVAILABLE:
                print("⚠️ VibeVoice package structure mismatch. Trying import from vibevoice...")
                from vibevoice import VibeVoice as VibeVoiceForConditionalGeneration

            # Check for downloaded model
            model_id = f"vibevoice-{self.model_size}" # e.g. vibevoice-7b
            model_path = model_manager.get_model_path(model_id)
            
            load_path = model_path if model_path else "vibevoice/VibeVoice-7B"
            if model_path:
                print(f"📂 Using local model from: {model_path}")
            else:
                print(f"☁️ Using HuggingFace Hub model: {load_path}")
            
            # Run blocking load in executor to avoid blocking asyncio loop
            loop = asyncio.get_running_loop()
            
            # Load Processor
            from transformers import AutoProcessor
            def _load_processor():
                try:
                    # Try AutoProcessor first
                    return AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
                except Exception:
                    # Fallback to manual import
                    print(f"⚠️ AutoProcessor failed. Importing VibeVoiceProcessor manually.")
                    try:
                        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
                        return VibeVoiceProcessor.from_pretrained(load_path, trust_remote_code=True)
                    except ImportError:
                         # Try another known path
                         from vibevoice.processor import VibeVoiceProcessor
                         return VibeVoiceProcessor.from_pretrained(load_path, trust_remote_code=True)

            self._processor = await loop.run_in_executor(None, _load_processor)

            def _load_model():
                # Force VRAM cleanup before loading to ensure space
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                # User prefers full precision / FP16 over 4-bit if VRAM allows (24GB fits 7B FP16)
                try:
                    print(f"🚀 Loading {self.display_name} in float16 (High Quality)...")
                    return VibeVoiceForConditionalGeneration.from_pretrained(
                        load_path, 
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                except ImportError:
                    # Fallback
                    return VibeVoiceForConditionalGeneration.from_pretrained(load_path, trust_remote_code=True).to("cuda")
                except Exception as e:
                    print(f"⚠️ FP16 load failed, trying memory efficient 4-bit fallback: {e}")
                    try:
                        from transformers import BitsAndBytesConfig
                        q_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16
                        )
                        return VibeVoiceForConditionalGeneration.from_pretrained(
                            load_path,
                            trust_remote_code=True,
                            quantization_config=q_config,
                            device_map="auto"
                        )
                    except Exception as e2:
                        raise RuntimeError(f"Failed to load model: {e2}") from e2

            self._model = await loop.run_in_executor(None, _load_model)
            
            self._initialized = True
            print(f"✅ {self.display_name} initialized successfully")
            return True
            
        except ImportError as e:
            print(f"⚠️ {self.display_name} dependency missing: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize {self.display_name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the engine and release GPU resources."""
        if self._model:
            del self._model
            self._model = None
            
            # Try to force garbage collection
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self._cloned_voices.clear()
        self._initialized = False
        print(f"🔓 {self.display_name} shutdown")
    
    def get_available_voices(self) -> List[VoiceInfo]:
        """Get available VibeVoice speakers plus any cloned voices."""
        voices = VIBEVOICE_SPEAKERS.copy()
        
        # Add cloned voices
        for voice_id in self._cloned_voices:
            voices.append(VoiceInfo(
                id=voice_id,
                name=f"Cloned: {voice_id}",
                gender="neutral",
                description="Zero-shot cloned voice",
                tags=["cloned", "zero-shot"]
            ))
        
        return voices
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "speaker_0",
        temperature: float = 0.7,
        top_p: float = 0.95,
        reference_audio: Optional[bytes] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate speech using VibeVoice."""
        if not self._initialized or not self._model:
            success = await self.initialize()
            if not success:
                raise RuntimeError(f"{self.display_name} is not available (GPU required)")
        
        try:
            # Prepare reference audio if needed
            if reference_audio:
                # TODO: Implement reference audio handling for library
                pass
            elif voice in self._cloned_voices:
                # TODO: Retrieve cached clone
                pass
            
            # Run generation in executor
            loop = asyncio.get_running_loop()
            
            # Define wrapper for blocking call
            def _generate():
                # Extract speaker ID and handle Zero Shot cloning
                speaker_id = 0
                voice_samples = None
                
                # Check if voice is a file path (Zero Shot)
                import os
                if isinstance(voice, str) and os.path.exists(voice) and os.path.isfile(voice):
                    # Zero Shot Mode
                    print(f"🎤 VibeVoice: Zero-Shot Cloning from {voice}")
                    voice_samples = [voice] # Mapped to Speaker 0
                    speaker_id = 0
                
                # Else check for speaker_ID format
                elif isinstance(voice, str) and voice.startswith("speaker_"):
                    try:
                        speaker_id = int(voice.split("_")[1])
                    except ValueError:
                        pass
                
                # Format text as script
                if voice_samples:
                    if not text.strip().startswith("Speaker"):
                        script_text = f"Speaker 1: {text}"
                    else:
                        script_text = text
                else:
                    script_text = f"Speaker {speaker_id}: {text}"
                
                # Process text inputs
                default_voice_paths = [
                    "/app/models/tts/xtts-v2/samples/en_sample.wav",
                    "/app/static_files/voice_samples/zac_sample.wav",
                    "/app/audio_samples/default_voice.wav"
                ]
                
                if voice_samples:
                    inputs = self._processor(text=script_text, voice_samples=voice_samples, return_tensors="pt")
                else:
                    fallback_voice = None
                    for path in default_voice_paths:
                        if os.path.exists(path):
                            fallback_voice = path
                            break
                    
                    if fallback_voice:
                        inputs = self._processor(text=script_text, voice_samples=[fallback_voice], return_tensors="pt")
                    else:
                        inputs = self._processor(text=script_text, return_tensors="pt")

                # Move to GPU safely
                import torch
                cuda_inputs = {}
                for k, v in inputs.items():
                    if v is not None and hasattr(v, "to"):
                        cuda_inputs[k] = v.to("cuda")
                
                # Generate
                return self._model.generate(
                    **cuda_inputs,
                    tokenizer=self._processor.tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    cfg_scale=1.3
                )
            
            audio_output = await loop.run_in_executor(None, _generate)
            
            # Extract audio
            import torch
            import numpy as np
            
            if hasattr(audio_output, "speech_outputs") and audio_output.speech_outputs:
                speech_list = audio_output.speech_outputs
                if isinstance(speech_list, list) and len(speech_list) > 0:
                    audio_tensors = [s.cpu() if hasattr(s, "cpu") else torch.tensor(s) for s in speech_list]
                    audio_array = torch.cat(audio_tensors, dim=-1).numpy()
                else:
                    raise RuntimeError("VibeVoice returned empty speech_outputs")
            elif hasattr(audio_output, "cpu"):
                # Fallback: direct tensor output
                audio_array = audio_output.cpu().numpy()
            elif isinstance(audio_output, list):
                audio_array = np.array(audio_output)
            else:
                # Try index access (ModelOutput)
                audio_array = audio_output[0].cpu().numpy()

            # Squeeze if needed (1, T) -> (T,) or (1, 1, T) -> (T,)
            while len(audio_array.shape) > 1 and audio_array.shape[0] == 1:
                audio_array = audio_array.squeeze(0)
            
            # Convert numpy/tensor to bytes (WAV)
            import io
            import wave
            
            # Scale if float -1..1 => int16
            if audio_array.dtype.kind == 'f':
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_array.tobytes())
            
            return GenerationResult(
                audio_data=wav_buffer.getvalue(),
                sample_rate=24000,
                voice_id=voice,
                metadata={
                    "engine": self.name,
                    "model": self.model_size
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"VibeVoice generation failed: {e}") from e
    
    async def clone_voice(
        self,
        audio_sample: bytes,
        voice_name: str,
        **kwargs
    ) -> VoiceInfo:
        """Clone a voice for zero-shot TTS."""
        self._cloned_voices[voice_name] = audio_sample
        return VoiceInfo(
            id=voice_name,
            name=f"Cloned: {voice_name}",
            gender="neutral",
            description="Zero-shot cloned voice",
            tags=["cloned", "zero-shot", "vibevoice"]
        )
    
    async def health_check(self) -> bool:
        """Check if engine is ready."""
        return self._initialized and self._model is not None
    
    def get_default_voice(self) -> str:
        return "speaker_0"
