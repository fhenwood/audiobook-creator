
"""
Maya1 TTS Engine

State-of-the-art expressive TTS model by Maya Research.
Uses Llama-style 3B model with SNAC audio compression.
"""

import os
import asyncio
import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union

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

# Configure logging
logger = logging.getLogger(__name__)

# Constants from Maya1 reference implementation
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
SNAC_TOKENS_PER_FRAME = 7

# Logits Processor to force Audio generation
from transformers import LogitsProcessor, LogitsProcessorList

class ForceSnacLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # Check if SOS token (128257) is in the sequence
        # We look at the last few tokens to be efficient, or scan whole sequence
        # input_ids[0] is the current batch sequence
        
        row = input_ids[0]
        
        # If SOS is NOT present, do not constrain (allow model to generate SOS)
        if CODE_START_TOKEN_ID not in row:
             return scores
             
        # If SOS IS present, strict masking for SNAC codes + EOS
        mask = torch.full_like(scores, float('-inf'))
        mask[:, SNAC_MIN_ID:SNAC_MAX_ID + 1] = 0
        mask[:, CODE_END_TOKEN_ID] = 0
        return scores + mask

# Default Voices with detailed descriptions
MAYA_VOICES = [
    VoiceInfo(
        id="maya_male_professional",
        name="Maya Professional (Male)",
        gender="male",
        description="Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing, neutral tone delivery at med intensity.",
        tags=["realistic", "professional", "podcast", "american"]
    ),
    VoiceInfo(
        id="maya_female_professional",
        name="Maya Professional (Female)",
        gender="female",
        description="Realistic female voice in the 30s age with american accent. Normal pitch, bright timbre, conversational pacing, neutral tone delivery at med intensity.",
        tags=["realistic", "professional", "podcast", "american"]
    ),
    VoiceInfo(
        id="maya_storyteller_male",
        name="Maya Storyteller (Male)",
        gender="male",
        description="Realistic male voice in the 40s age with british accent. Deep warmth timbre, slow measured pacing, storytelling tone delivery at low intensity.",
        tags=["storyteller", "british", "warm"]
    ),
    VoiceInfo(
        id="maya_villain",
        name="Maya Villain",
        gender="male",
        description="Creative, dark_villain character. Male voice in their 40s with british accent. Low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
        tags=["creative", "villain", "dark"]
    ),
    VoiceInfo(
        id="maya_robot",
        name="Maya Robot",
        gender="neutral",
        description="Creative, ai_machine_voice character. Monotone pitch, metallic timbre, constant pacing, robotic tone delivery.",
        tags=["creative", "robotic"]
    ),
]

@register_engine
class MayaEngine(TTSEngine):
    """
    Maya1 TTS Engine (3B parameters).
    """
    
    name = "maya"
    display_name = "Maya1 (3B)"
    version = "1.0.0"
    
    capabilities = [
        EngineCapability.EMOTION_CONTROL,
        EngineCapability.MULTI_SPEAKER,  # Via textual descriptions
        EngineCapability.LONG_FORM,
    ]
    
    min_vram_gb = 8.0
    recommended_vram_gb = 16.0
    
    def __init__(self, config: Optional[EngineConfig] = None):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._snac = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self) -> bool:
        """Initialize Maya1 engine."""
        try:
            # CPU/GPU Check
            if not torch.cuda.is_available():
                logger.warning(f"⚠️ {self.display_name} performs best on GPU. CPU generation will be very slow.")
                self._device = "cpu"
                
            logger.info(f"🔄 Loading {self.display_name}...")
            
            # Acquire GPU resources
            if self._device == "cuda":
                success, msg = gpu_manager.acquire_maya()
                if not success:
                    logger.error(f"❌ Failed to acquire GPU for {self.display_name}: {msg}")
                    return False
            
            # Imports inside method to avoid heavy load at startup
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from snac import SNAC
            
            # Check for downloaded model
            model_id = "maya-1"
            model_path = model_manager.get_model_path(model_id)
            if not model_path:
                logger.info("☁️ Model not found locally. It will be downloaded by transformers.")
                model_path = "maya-research/Maya1"
            else:
                logger.info(f"📂 Using local model from: {model_path}")

            # Run loading in executor
            loop = asyncio.get_running_loop()
            
            def _load():
                # Load Tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # Load Model
                # Use bfloat16 for efficiency if on GPU
                torch_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self._device == "cuda" else None,
                    trust_remote_code=True
                )
                if self._device == "cpu":
                    model = model.to("cpu")
                
                # Load SNAC
                logger.info(f"🎵 Loading SNAC decoder...")
                snac = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(self._device)
                
                return tokenizer, model, snac
            
            self._tokenizer, self._model, self._snac = await loop.run_in_executor(None, _load)
            
            self._initialized = True
            logger.info(f"✅ {self.display_name} initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"⚠️ {self.display_name} dependency missing: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.display_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        if self._model:
            del self._model
            del self._tokenizer
            del self._snac
            self._model = None
            self._tokenizer = None
            self._snac = None
            
            # Cleanup GPU
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self._initialized = False
        logger.info(f"🔓 {self.display_name} shutdown")

    def get_available_voices(self) -> List[VoiceInfo]:
        return MAYA_VOICES
    
    def get_default_voice(self) -> str:
        return "maya_male_professional"

    def _unpack_snac_from_7(self, vocab_ids: List[int]) -> List[List[int]]:
        """Unpack 7-token SNAC frames to 3 hierarchical levels."""
        # Remove EOS token if present
        if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
            vocab_ids = vocab_ids[:-1]
        
        # Ensure complete frames
        frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
        vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            slots = vocab_ids[i*7:(i+1)*7]
            
            # Subtract offset and mod 4096 to get original SNAC codes
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,  # Even
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,  # Odd
            ])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]

    def _decode_snac(self, snac_tokens: List[int]) -> bytes:
        """Decode SNAC tokens to audio bytes."""
        levels = self._unpack_snac_from_7(snac_tokens)
        
        if not levels[0]:
            return b""
        
        # Convert to tensors
        codes = [
            torch.tensor(level, dtype=torch.long, device=self._device).unsqueeze(0)
            for level in levels
        ]
        
        # Decode
        with torch.no_grad():
            z_q = self._snac.quantizer.from_codes(codes)
            audio = self._snac.decoder(z_q)
        
        # Extract audio: [batch, 1, samples] → [samples]
        audio = audio[0, 0].cpu().numpy()
        
        # Convert float32 to int16 PCM
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def generate_speech(
        self,
        text: str,
        voice: str = "maya_male_professional",
        temperature: float = 0.4,
        top_p: float = 0.9,
        voice_description: str = None,
        **kwargs
    ) -> GenerationResult:
        """Generate speech using Maya1."""
        if not self._initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError(f"{self.display_name} failed to initialize")

        # Resolve voice description
        voice_desc = None
        if voice_description:
            voice_desc = voice_description
        else:
            for v in MAYA_VOICES:
                if v.id == voice:
                    voice_desc = v.description + " " + " ".join(v.tags)
                    break
        
        # Fallback if voice ID not found or if custom description passed (e.g. as voice arg)
        if not voice_desc:
            voice_desc = voice # Assume the voice argument IS the description if not a registered ID

        # Build prompt using chat template
        # Manually construct prompt to avoid Llama3 default system message injection
        # Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        
        content = f'<description="{voice_desc}"> {text}'
        
        # Use tokenizer to encode specific special tokens if possible, or construct string
        # We'll construct the raw string as it's cleaner for Llama 3
        
        prompt_str = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{content}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        prompt_ids = self._tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(self._device)

        # Append SOS token to trigger audio generation
        sos_token = torch.tensor([[CODE_START_TOKEN_ID]], device=self._device)
        prompt_ids = torch.cat([prompt_ids, sos_token], dim=1)
        
        # Run generation logic in executor
        loop = asyncio.get_running_loop()
        
        def _generate():
            logger.info("Starting generation inference...")
            
            # Use LogitsProcessor to force audio codes
            logits_processor = LogitsProcessorList([ForceSnacLogitsProcessor()])
            
            with torch.no_grad():
                output_ids = self._model.generate(
                    prompt_ids,
                    max_new_tokens=2000, 
                    min_new_tokens=50, # Force some generation
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=CODE_END_TOKEN_ID,
                    repetition_penalty=1.1,
                    logits_processor=logits_processor
                )
            
            logger.info(f"Generation output shape: {output_ids.shape}")
            
            # Extract generated part
            generated_ids = output_ids[0][prompt_ids.shape[1]:]
            logger.info(f"Generated {len(generated_ids)} new tokens")
            
            # Debug: print first few tokens to see what they are
            logger.info(f"First 20 generated tokens: {generated_ids[:20].tolist()}")
            
            # Filter for SNAC codes
            snac_tokens = [
                token_id.item() 
                for token_id in generated_ids 
                if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
            ]
            logger.info(f"Filtered SNAC tokens: {len(snac_tokens)}")
            
            if len(snac_tokens) < 10:
                logger.warning("⚠️  Very few SNAC tokens! Model might be outputting garbage.")

            return self._decode_snac(snac_tokens)

        audio_bytes = await loop.run_in_executor(None, _generate)
        
        # Convert raw PCM to WAV
        import io
        import wave
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1) # Mono
            wav_file.setsampwidth(2) # 16-bit
            wav_file.setframerate(24000) # SNAC standard
            wav_file.writeframes(audio_bytes)
            
        wav_data = wav_buffer.getvalue()
        
        return GenerationResult(
            audio_data=wav_data,
            sample_rate=24000,
            voice_id=voice,
            metadata={
                "engine": self.name,
                "model": "maya-1"
            }
        )

    def unload(self):
        """Unload model from GPU to free resources."""
        if self._model is not None:
            logger.info("🗑️ Unloading Maya1 model from GPU...")
            del self._model
            if self._tokenizer:
                del self._tokenizer
            if self._snac:
                del self._snac
            self._model = None
            self._tokenizer = None
            self._snac = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✅ Maya1 unloaded.")
        self._initialized = False

    def _pack_snac_to_7(self, codes: List[List[int]]) -> List[int]:
        """Pack 3 hierarchical SNAC levels into 7-token frames."""
        l1, l2, l3 = codes
        
        if not l1:
            return []
        
        frames = len(l1)
        vocab_ids = []
        
        for i in range(frames):
            slot0 = (l1[i] % 4096) + CODE_TOKEN_OFFSET
            slot1 = (l2[i*2] % 4096) + CODE_TOKEN_OFFSET
            slot2 = (l3[i*4] % 4096) + CODE_TOKEN_OFFSET
            slot3 = (l3[i*4 + 1] % 4096) + CODE_TOKEN_OFFSET
            slot4 = (l2[i*2 + 1] % 4096) + CODE_TOKEN_OFFSET
            slot5 = (l3[i*4 + 2] % 4096) + CODE_TOKEN_OFFSET
            slot6 = (l3[i*4 + 3] % 4096) + CODE_TOKEN_OFFSET
            
            vocab_ids.extend([slot0, slot1, slot2, slot3, slot4, slot5, slot6])
        
        return vocab_ids

    async def encode_audio_to_snac(self, audio_path: str) -> List[int]:
        """Encode audio file to SNAC token vocabulary IDs."""
        if not self._initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError(f"{self.display_name} failed to initialize")
        
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=24000, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self._device)
        
        with torch.no_grad():
            codes = self._snac.encode(audio_tensor)
        
        # Debug: log the structure of SNAC codes
        logger.info(f"SNAC encode returned {len(codes)} tensors")
        for i, c in enumerate(codes):
            logger.info(f"  codes[{i}] shape: {c.shape}")
        
        # SNAC returns list of tensors with shape [batch, codebook, seq_len]
        # For snac_24khz: 3 tensors with 1 codebook each
        l1 = codes[0].squeeze().cpu().tolist()  # [seq_len]
        l2 = codes[1].squeeze().cpu().tolist()  # [seq_len * 2] 
        l3 = codes[2].squeeze().cpu().tolist()  # [seq_len * 4]
        
        # Handle case where squeeze returns a scalar (single frame)
        if isinstance(l1, int):
            l1, l2, l3 = [l1], [l2], [l3]
        
        vocab_ids = self._pack_snac_to_7([l1, l2, l3])
        logger.info(f"🎵 Encoded {len(audio)/sr:.2f}s audio to {len(vocab_ids)} SNAC tokens")
        return vocab_ids

    async def generate_with_reference(
        self,
        text: str,
        reference_audio_path: str,
        reference_transcript: str,
        voice_description: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """
        Generate speech with voice characteristics from a reference.
        
        Maya doesn't support direct SNAC voice cloning - instead, we use
        the voice description (from Qwen2-Audio analysis) to condition
        the generation for a similar voice style.
        
        Args:
            text: Text to speak
            reference_audio_path: Reference audio (used for analysis if no description)
            reference_transcript: Transcript of reference (currently unused)
            voice_description: Voice description from Qwen analysis
        """
        if not voice_description:
            # If no description provided, we can't do much - use default voice
            logger.warning("No voice description provided. Using default Maya voice.")
            return await self.generate_speech(
                text=text,
                voice="maya_male_professional",
                temperature=temperature,
                top_p=top_p
            )
        
        # Use the analyzed voice description for generation
        logger.info(f"🎭 Generating with voice description: {voice_description[:100]}...")
        
        return await self.generate_speech(
            text=text,
            voice="custom",  # Custom voice via description
            voice_description=voice_description,
            temperature=temperature,
            top_p=top_p
        )


