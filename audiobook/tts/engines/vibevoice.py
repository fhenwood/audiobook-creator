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
        """Initialize VibeVoice engine via subprocess."""
        if self._initialized:
            return True
            
        print(f"🔄 Initializing {self.display_name} (Subprocess Mode)...")
        
        # CPU/GPU Check
        try:
             import torch
             if not torch.cuda.is_available():
                 print(f"⚠️ {self.display_name} requires CUDA GPU.")
                 return False
        except:
             return False

        # Acquire resources
        success, msg = gpu_manager.acquire_vibevoice()
        if not success:
            print(f"❌ Failed to acquire GPU: {msg}")
            return False

        # Setup Subprocess
        import multiprocessing as mp
        from audiobook.tts.engines.vibevoice_worker import run_worker, WorkerCommand, WorkerResult
        
        try:
            # Use 'spawn' for CUDA compatibility
            ctx = mp.get_context('spawn')
            self._cmd_q = ctx.Queue()
            self._res_q = ctx.Queue()
            
            self._process = ctx.Process(
                target=run_worker,
                args=(self._cmd_q, self._res_q, self.model_size),
                daemon=True
            )
            self._process.start()
            print(f"🚀 VibeVoice Worker started (PID: {self._process.pid})")
            
            # Send Init Command
            loop = asyncio.get_running_loop()
            
            def _send_init():
                self._cmd_q.put(WorkerCommand("INIT"))
                return self._res_q.get(timeout=60) # High timeout for model load
                
            result = await loop.run_in_executor(None, _send_init)
            
            if result.success:
                self._initialized = True
                print(f"✅ {self.display_name} initialized successfully")
                return True
            else:
                print(f"❌ Worker init failed: {result.error}")
                self._kill_process()
                return False
                
        except Exception as e:
            print(f"❌ Failed to start worker: {e}")
            self._kill_process()
            return False
    
    async def shutdown(self) -> None:
        """Shutdown worker process to force VRAM release."""
        print(f"🛑 Shutting down VibeVoice worker...")
        
        # Lazy import
        from audiobook.utils.gpu_resource_manager import gpu_manager
        gpu_manager.log_gpu_stats("VibeVoice Pre-Shutdown")
        
        self._kill_process()
        
        self._initialized = False
        gpu_manager.log_gpu_stats("VibeVoice Post-Shutdown")
        
        # Double check VRAM releases
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"✅ VibeVoice shutdown complete (Process Terminated)")

    def _kill_process(self):
        """Helper to terminate process safely."""
        from audiobook.tts.engines.vibevoice_worker import WorkerCommand
        import queue
        
        if hasattr(self, "_process") and self._process and self._process.is_alive():
            try:
                # Try graceful first
                if hasattr(self, "_cmd_q"):
                    self._cmd_q.put(WorkerCommand("SHUTDOWN"))
                    self._process.join(timeout=3)
            except:
                pass
            
            # Force kill if still alive
            if self._process.is_alive():
                print("⚠️ Force killing worker process...")
                self._process.terminate()
                self._process.join(timeout=1)
                
                if self._process.is_alive():
                    self._process.kill()
                    
        self._process = None
        self._cmd_q = None
        self._res_q = None

    def get_default_voice(self) -> str:
        return "speaker_0"

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
        """Generate speech via worker process."""
        print(f"DEBUG: VibeVoice generate_speech called with voice={voice}")
        print(f"DEBUG: reference_audio type: {type(reference_audio)}")
        if reference_audio:
            if isinstance(reference_audio, str):
                 print(f"DEBUG: reference_audio path: {reference_audio}")
            elif isinstance(reference_audio, bytes):
                 print(f"DEBUG: reference_audio bytes len: {len(reference_audio)}")
        
        # Check if process is alive
        if not self._initialized or not hasattr(self, "_process") or not self._process or not self._process.is_alive():
            success = await self.initialize()
            if not success:
                raise RuntimeError("Service unavailable - could not start worker")
                
        # Smart detection: If voice is a file path and reference_audio is missing, use voice as reference
        if reference_audio is None and isinstance(voice, str) and (voice.endswith(".wav") or voice.endswith(".mp3") or "/" in voice):
            if os.path.exists(voice):
                print(f"DEBUG: Detected voice argument is path, using as reference_audio: {voice}")
                reference_audio = voice
                
        try:
            loop = asyncio.get_running_loop()
            
            payload = {
                "text": text,
                "voice": voice,
                "temperature": temperature,
                "top_p": top_p,
                "reference_audio": reference_audio
            }
            
            # Helper to blocking send/recv
            from audiobook.tts.engines.vibevoice_worker import WorkerCommand
            def _do_work():
                self._cmd_q.put(WorkerCommand("GENERATE", payload))
                return self._res_q.get(timeout=300) # 5 min timeout for long form
            
            result = await loop.run_in_executor(None, _do_work)
            
            if not result.success:
                raise RuntimeError(f"Worker Error: {result.error}")
                
            return GenerationResult(
                audio_data=result.data,
                sample_rate=24000,
                voice_id=voice,
                metadata={"engine": self.name, "mode": "subprocess"}
            )
            
        except Exception as e:
            raise RuntimeError(f"VibeVoice generation failed: {e}")
            
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
        return self._initialized and self._process and self._process.is_alive()
