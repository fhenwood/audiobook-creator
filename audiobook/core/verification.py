
import os
import difflib
from typing import Tuple, Optional
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)

class TranscriptionVerifier:
    """
    Verifies audio against text using Whisper transcription.
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "float32"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        
    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
                self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                logger.info("Whisper model loaded.")
            except ImportError:
                logger.error("faster-whisper not installed. Verification disabled.")
                raise RuntimeError("faster-whisper not installed")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

    def verify(self, audio_path: str, expected_text: str, threshold: float = 0.8) -> Tuple[bool, float, str]:
        """
        Verify if audio content matches expected text.
        
        Args:
            audio_path: Path to audio file.
            expected_text: Text that should be in the audio.
            threshold: Similarity threshold (0.0 to 1.0). Default 0.8 (80%).
            
        Returns:
            (passed, score, transcribed_text)
        """
        if not os.path.exists(audio_path):
            return False, 0.0, ""
            
        self._load_model()
        
        try:
            segments, _ = self._model.transcribe(audio_path, beam_size=5)
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            # Normalize for comparison
            def normalize(s):
                return ' '.join(s.lower().split())
            
            norm_expected = normalize(expected_text)
            norm_transcribed = normalize(transcribed_text)
            
            if not norm_expected and not norm_transcribed:
                return True, 1.0, ""
            if not norm_expected:
                return False, 0.0, transcribed_text
                
            matcher = difflib.SequenceMatcher(None, norm_expected, norm_transcribed)
            score = matcher.ratio()
            
            passed = score >= threshold
            
            if not passed:
                logger.info(f"Mismatch: Score {score:.2f} < {threshold:.2f}")
                logger.info(f"Expected: \"{expected_text}\"")
                logger.info(f"Got:      \"{transcribed_text}\"")
            
            return passed, score, transcribed_text
            
        except Exception as e:
            logger.error(f"Verification error for {audio_path}: {e}")
            return False, 0.0, f"Error: {str(e)}"

    def unload(self):
        """Unload model to free memory."""
        if self._model:
            del self._model
            self._model = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded.")
