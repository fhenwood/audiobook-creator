
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

async def test_style(voice_id, text):
    engine_id = "maya"
    logger.info(f"========== Testing Voice Style: {voice_id} ==========")
    
    try:
        logger.info(f"Generating audio...")
        result = await tts_service.generate(text, engine=engine_id, voice=voice_id)
        
        if not result or not result.audio_data:
            logger.error("❌ Generation failed: No audio data returned.")
            return False

        # Save to file
        output_dir = "tests/output/styles"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{voice_id}.wav"
        path = os.path.join(output_dir, filename)
        
        with open(path, "wb") as f:
            f.write(result.audio_data)
            
        logger.info(f"✅ SUCCESS: Saved to {path} ({len(result.audio_data)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return False

async def main():
    # Test Villain and Robot
    styles = [
        ("maya_villain", "I am the darkness that creeps in the night."),
        ("maya_robot", "System initialized. Target acquired.")
    ]
    
    for voice_id, text in styles:
        await test_style(voice_id, text)

if __name__ == "__main__":
    asyncio.run(main())
