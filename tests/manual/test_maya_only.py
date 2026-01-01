
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

async def test_maya():
    engine_id = "maya"
    voice = "maya_male_professional"
    text = "Hello! This is a test of the Maya One Text to Speech system."
    
    logger.info(f"========== Testing Engine: {engine_id} ==========")
    
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
    
    logger.info("Starting Maya1 E2E Test...")
    success = await test_maya()
    
    if success:
        logger.info("🎉 MAYA TEST PASSED")
        sys.exit(0)
    else:
        logger.error("⚠️ MAYA TEST FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
