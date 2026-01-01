"""
Line Generator - Handles TTS generation for individual lines.

Extracts line generation logic from generator.py for modularity and testability.
"""

import os
import asyncio
import tempfile
from typing import Optional, Dict, Any, Tuple
from pydub import AudioSegment
import io
from audiobook.tts.service import tts_service
from audiobook.tts.generator.utils import is_only_punctuation, split_and_annotate_text
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # seconds


class LineGenerator:
    """
    Generates TTS audio for individual lines.
    
    Handles:
    - Single voice mode
    - Dialogue splitting with separate voices
    - Retry logic for failed generations
    - Concurrency control via semaphore
    """
    
    def __init__(
        self,
        engine: str,
        narrator_voice: str,
        dialogue_voice: Optional[str] = None,
        use_dialogue_split: bool = False,
        concurrency: int = 8,
        verifier: Optional[Any] = None,
        **tts_kwargs
    ):
        self.engine = engine.lower()
        self.narrator_voice = narrator_voice
        self.dialogue_voice = dialogue_voice or narrator_voice
        self.use_dialogue_split = use_dialogue_split
        self.tts_kwargs = tts_kwargs
        self._semaphore = asyncio.Semaphore(concurrency)
        self.verifier = verifier
    
    async def generate_line(
        self,
        line_index: int,
        line: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Generate audio for a single line.
        
        Args:
            line_index: Index of the line in the book.
            line: Text content to convert.
            output_path: Path to save the audio file.
            
        Returns:
            Dict with generation result metadata.
        """
        if not line or is_only_punctuation(line):
            return {"index": line_index, "skipped": True, "reason": "empty_or_punctuation"}
        
        try:
            if self.use_dialogue_split:
                await self._generate_with_dialogue_split(line, output_path)
            else:
                await self._generate_single_voice(line, output_path)
            
            return {"index": line_index, "success": True}
            
        except Exception as e:
            logger.error(f"Failed to generate line {line_index}: {e}")
            return {"index": line_index, "failed": True, "error": str(e)}
    
    async def _generate_single_voice(self, text: str, output_path: str):
        """Generate audio using single voice."""
        text_clean = text.replace('"', '').replace('\\', '').strip()
        
        if not text_clean or is_only_punctuation(text_clean):
            return
        
        audio_data = await self._call_tts_with_retry(text_clean, self.narrator_voice)
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
    
    async def _generate_with_dialogue_split(self, text: str, output_path: str):
        """Generate audio with separate voices for dialogue and narration."""
        parts = split_and_annotate_text(text)
        combined_audio = AudioSegment.empty()
        
        for part in parts:
            part_text = part["text"].strip().replace('"', '').replace('\\', '')
            
            if not part_text or is_only_punctuation(part_text):
                continue
            
            voice = self.dialogue_voice if part["type"] == "dialogue" else self.narrator_voice
            
            # Generate to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                audio_data = await self._call_tts_with_retry(part_text, voice)
                with open(temp_path, "wb") as f:
                    f.write(audio_data)
                
                part_segment = AudioSegment.from_wav(temp_path)
                combined_audio += part_segment
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        if len(combined_audio) > 0:
            combined_audio.export(output_path, format="wav")
    
    async def _call_tts_with_retry(self, text: str, voice: str) -> bytes:
        """Call TTS service with retry logic."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    result = await tts_service.generate(
                        text=text,
                        voice=voice,
                        engine=self.engine,
                        **self.tts_kwargs
                    )
                    
                    # Verify if verifier is present
                    if self.verifier:
                        await self._verify_audio(result.audio_data, text)
                        
                    return result.audio_data

            except Exception as e:
                last_error = e
                delay = RETRY_DELAY_BASE * (attempt + 1)
                logger.warning(f"TTS attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        raise last_error if last_error else Exception("Unknown TTS error")

    async def _verify_audio(self, audio_data: bytes, text: str):
        """Verify audio matches text."""
        if not self.verifier:
            return
            
        # Write bytes to temp file for faster-whisper (it needs path or file-like)
        # Using temp file is safer for now as faster-whisper might expect real file for some backends
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
            
        try:
            passed, score, transcribed = self.verifier.verify(tmp_path, text)
            if not passed:
                raise ValueError(f"Verification failed: score {score:.2f} < threshold. Got: '{transcribed[:50]}...'")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

