"""
Orpheus TTS - FastAPI Server
Based on Lex-au/Orpheus-FastAPI (Apache 2.0 License)

Main FastAPI server for Orpheus Text-to-Speech with OpenAI-compatible API.
"""

import os
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Create default .env if it doesn't exist
def ensure_env_file_exists():
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            default_env = {}
            with open(".env.example", "r") as example_file:
                for line in example_file:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        default_env[key] = line.split("=", 1)[1].strip()

            final_env = default_env.copy()
            for key in default_env:
                if key in os.environ:
                    final_env[key] = os.environ[key]

            with open(".env", "w") as env_file:
                for key, value in final_env.items():
                    env_file.write(f"{key}={value}\n")
            print("✅ Created default .env file")
        except Exception as e:
            print(f"⚠️ Error creating default .env file: {e}")

ensure_env_file_exists()
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tts_engine import (
    generate_speech_from_api, 
    AVAILABLE_VOICES, 
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES
)

app = FastAPI(
    title="Orpheus-TTS",
    description="High-performance Text-to-Speech server with OpenAI-compatible API",
    version="1.0.0"
)

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)

# Mount outputs directory
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


class APIResponse(BaseModel):
    status: str
    voice: str
    output_file: str
    generation_time: float


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "orpheus-tts", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    use_batching = len(request.input) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(request.input)} characters)")
    
    start = time.time()
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
    )


@app.get("/v1/audio/voices")
async def list_voices():
    """Return list of available voices"""
    if not AVAILABLE_VOICES or len(AVAILABLE_VOICES) == 0:
        raise HTTPException(status_code=404, detail="No voices available")
    return JSONResponse(
        content={
            "status": "ok",
            "voices": AVAILABLE_VOICES,
            "default": DEFAULT_VOICE,
            "voice_languages": VOICE_TO_LANGUAGE,
            "languages": AVAILABLE_LANGUAGES
        }
    )


@app.post("/speak")
async def speak(request: dict):
    """Legacy endpoint for compatibility"""
    text = request.get("text", "")
    voice = request.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing 'text'"}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    use_batching = len(text) > 1000
    
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=1000
    )
    end = time.time()
    generation_time = round(end - start, 2)

    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": output_path,
        "generation_time": generation_time
    })


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("ORPHEUS_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("ORPHEUS_PORT", "8880"))
    except (ValueError, TypeError):
        port = 8880
    
    print(f"🔥 Starting Orpheus-TTS Server on {host}:{port}")
    print(f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    api_url = os.environ.get("ORPHEUS_API_URL")
    if not api_url:
        print("⚠️ ORPHEUS_API_URL not set. Please configure in .env file.")
    else:
        print(f"🔗 Using LLM inference server at: {api_url}")
    
    uvicorn.run("app:app", host=host, port=port, reload=False)
