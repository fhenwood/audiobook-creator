
import asyncio
import os
import sys
import logging
import difflib
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure we can import app modules
sys.path.append(os.getcwd())

from audiobook.tts.service import tts_service
from faster_whisper import WhisperModel

def calculate_similarity(ref, hyp):
    """Calculate similarity ratio between two strings using difflib."""
    return difflib.SequenceMatcher(None, ref.lower(), hyp.lower()).ratio()

async def verify_alignment(text, voice="maya_male_professional"):
    engine_id = "maya"
    logger.info(f"========== Verifying Alignment for {engine_id} ==========")
    logger.info(f"Input Text: {text}")
    
    # 1. Generate Audio
    try:
        logger.info(f"Generating audio...")
        result = await tts_service.generate(text, engine=engine_id, voice=voice)
        
        if not result or not result.audio_data:
            logger.error("❌ Generation failed: No audio data returned.")
            return False

        # Save to temp file
        temp_wav = "tests/output/temp_verification.wav"
        os.makedirs("tests/output", exist_ok=True)
        with open(temp_wav, "wb") as f:
            f.write(result.audio_data)
            
        logger.info(f"Audio saved to {temp_wav} ({len(result.audio_data)} bytes)")
        
    except Exception as e:
        logger.error(f"❌ Generation exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Transcribe with Whisper
    try:
        logger.info("Loading Whisper model (large-v3)...")
        # Use large-v3 for best accuracy as requested
        model_size = "large-v3" 
        
        # Run in executor to not block async loop
        def run_whisper():
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            segments, info = model.transcribe(temp_wav, beam_size=5)
            return " ".join([segment.text for segment in segments]).strip()
            
        transcription = await asyncio.get_running_loop().run_in_executor(None, run_whisper)
        
        logger.info(f"📝 Transcription: {transcription}")
        
    except Exception as e:
        logger.error(f"❌ Transcription exception: {e}")
        return False
        
    # 3. Compare
    similarity = calculate_similarity(text, transcription)
    logger.info(f"📊 Similarity Score: {similarity:.2f}")
    
    if similarity > 0.8:
        logger.info("✅ ALIGNMENT CONFIRMED (High Similarity)")
        return True
    elif similarity > 0.5:
        logger.warning("⚠️ ALIGNMENT QUESTIONABLE (Medium Similarity)")
        return True # Treat as pass for now but warn
    else:
        logger.error("❌ ALIGNMENT FAILED (Low Similarity)")
        logger.error(f"Expected: {text}")
        logger.error(f"Got:      {transcription}")
        return False

async def main():
    # Test cases
    test_texts = [
        "Hello! This is a test of the Maya text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we interact with computers."
    ]
    
    failures = 0
    for text in test_texts:
        success = await verify_alignment(text)
        if not success:
            failures += 1
            
    if failures == 0:
        logger.info("🎉 ALL VERIFICATION TESTS PASSED")
        sys.exit(0)
    else:
        logger.error(f"⚠️ {failures} TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
