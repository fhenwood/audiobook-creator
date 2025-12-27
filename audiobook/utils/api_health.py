"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import traceback
from dotenv import load_dotenv

load_dotenv()

async def check_if_audio_generator_api_is_up(client):
    """Check if Orpheus TTS API is accessible and working."""
    try:
        async with client.audio.speech.with_streaming_response.create(
            model="orpheus",
            voice="tara",
            response_format="wav",
            speed=0.85,
            input="Hello, how are you?",
            timeout=120
        ) as response:
            return True, None
    except Exception as e:
        traceback.print_exc()
        return False, f"The Orpheus TTS API is not working. Please check if the TTS service is running and the .env file is correctly configured. Error: " + str(e)