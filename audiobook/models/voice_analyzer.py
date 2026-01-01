
import os
import torch
import logging
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from audiobook.models.manager import model_manager
from audiobook.utils.gpu_resource_manager import gpu_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAnalyzer:
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceAnalyzer, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        """Load the Qwen2-Audio model if not already loaded."""
        if self._model is not None:
            return

        model_id = "voice-analyzer-qwen"
        model_path = model_manager.get_model_path(model_id)
        
        if not model_path:
            logger.info(f"Model {model_id} not found locally. Initiating download...")
            model_path = model_manager.download_model(model_id)
            
        logger.info(f"🔄 Loading Voice Analyzer from {model_path}...")
        
        # Acquire GPU resources
        success, msg = gpu_manager.acquire_voice_analyzer()
        if not success:
            raise RuntimeError(f"Failed to acquire GPU resources: {msg}")

        # Load model and processor
        try:
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce VRAM (~8GB instead of ~15GB)
                trust_remote_code=True
            )
            logger.info("✅ Voice Analyzer loaded successfully (bfloat16)")
        except Exception as e:
            logger.error(f"Failed to load Voice Analyzer: {e}")
            raise

    def analyze_voice(self, audio_path: str) -> str:
        """
        Analyze the given audio file and return a detailed voice description.
        Targeting Maya1's prompt format: gender, age, accent, pitch, timbre, pacing, tone.
        """
        self._load_model()
        
        # Load audio properly
        import librosa
        audio_array, sr = librosa.load(audio_path, sr=16000)  # Qwen2-Audio expects 16kHz
        
        # Use proper ChatML format via apply_chat_template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": (
                        "Describe this voice in detail for a text-to-speech system. "
                        "Focus on: Gender, Estimated Age, Accent, Pitch (Low/Normal/High), "
                        "Timbre (e.g. Warm, Bright, Gravelly), Pacing (Slow/Conversational/Fast), "
                        "and emotional Tone. Be concise but specific."
                    )}
                ]
            }
        ]
        
        # Apply the chat template
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process with audio
        inputs = self._processor(text=text, audios=[audio_array], return_tensors="pt", padding=True).to("cuda")
        
        logger.info("Analyzing voice features...")
        logger.info(f"Input shape: {inputs.input_ids.shape}")
        
        generated_ids = self._model.generate(**inputs, max_new_tokens=256)
        
        logger.info(f"Output shape: {generated_ids.shape}")
        
        # Slice off input tokens to get only the generated part
        generated_ids_only = generated_ids[:, inputs.input_ids.size(1):]
        response = self._processor.batch_decode(generated_ids_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        logger.info(f"Analysis result: {response[:200]}")
        return response

    async def load(self):
        """Async wrapper for model loading."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)
    
    async def unload(self):
        """Unload model and free GPU memory."""
        if self._model is not None:
            logger.info("🗑️ Unloading Voice Analyzer...")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gpu_manager.release_voice_analyzer()
            logger.info("✅ Voice Analyzer unloaded")

voice_analyzer = VoiceAnalyzer()
