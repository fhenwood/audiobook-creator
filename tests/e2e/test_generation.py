import asyncio
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure we can import app modules
sys.path.append(os.getcwd())

from audiobook.tts.service import tts_service
from audiobook.models.manager import model_manager

async def ensure_model(model_id):
    """Ensure a model is downloaded and installed."""
    if model_manager.get_model_path(model_id):
        logger.info(f"Model {model_id} is already installed.")
        return True
    
    logger.info(f"Model {model_id} not found. Downloading (this may take a while)...")
    try:
        model_manager.download_model(model_id)
        logger.info(f"Model {model_id} downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        return False

async def test_engine(engine_id, model_id, text, voice=None):
    """Test generation for a specific engine."""
    logger.info(f"========== Testing Engine: {engine_id} ==========")
    
    if not await ensure_model(model_id):
        logger.error(f"Skipping test for {engine_id} due to missing model.")
        return False

    try:
        logger.info(f"Generating audio for {engine_id} (Voice: {voice})...")
        result = await tts_service.generate(text, engine=engine_id, voice=voice)
        
        if result and result.audio_data and len(result.audio_data) > 0:
            output_dir = "tests/output"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"test_{engine_id}.wav"
            path = os.path.join(output_dir, filename)
            
            with open(path, "wb") as f:
                f.write(result.audio_data)
            
            logger.info(f"✅ SUCCESS: Generated {len(result.audio_data)} bytes. Saved to {path}")
            return True
        else:
            logger.error(f"❌ FAILURE: Generation returned no audio data for {engine_id}.")
            return False
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION during generation for {engine_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    # create test output directory
    os.makedirs("tests/output", exist_ok=True)
    
    # Test VibeVoice
    logger.info("Starting VibeVoice E2E Test...")
    vibe_success = await test_engine(
        engine_id="vibevoice", 
        model_id="vibevoice-7b", 
        text="This is a test of the VibeVoice text to speech engine. It should sound high fidelity.",
        voice="speaker_0"
    )
    
    # Test Orpheus (XTTS)
    logger.info("Starting Orpheus (XTTS) E2E Test...")
    orpheus_success = await test_engine(
        engine_id="orpheus",
        model_id="xtts-v2",
        text="This is a test of the Orpheus engine. I am speaking with emotion.",
        voice="zac"
    )
    
    if vibe_success and orpheus_success:
        logger.info("🎉 ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("⚠️ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
