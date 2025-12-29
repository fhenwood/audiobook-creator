"""
REST API Layer for Audiobook Creator

This module provides the FastAPI-based REST API that enables:
- Multiple frontends (React, iOS, CLI, etc.)
- Programmatic access to audiobook generation
- Job management via API

Mount this alongside Gradio in the main app.
"""

import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import asyncio

from audiobook.tts.engines import list_engines, get_engine, VoiceInfo
from audiobook.tts.service import tts_service
from audiobook.utils.job_manager import job_manager
from audiobook.models.manager import model_manager, ModelStatus


# ============================================================================
# Request/Response Models
# ============================================================================

class EngineInfo(BaseModel):
    """Information about a TTS engine."""
    name: str
    display_name: str
    voice_count: int
    capabilities: List[str]
    min_vram_gb: float
    recommended_vram_gb: float


class VoiceInfoResponse(BaseModel):
    """Voice information."""
    id: str
    name: str
    gender: str
    description: str
    tags: List[str]


class GenerateRequest(BaseModel):
    """Request for TTS generation."""
    text: str
    engine: str = "orpheus"
    voice: Optional[str] = None
    emotion: Optional[str] = None
    speed: float = 1.0
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    """Request to create a new audiobook job."""
    title: str
    book_file_path: Optional[str] = None
    engine: str = "orpheus"
    narrator_voice: str = "zac"
    output_format: str = "m4b"
    use_emotion_tags: bool = True


class JobStatus(BaseModel):
    """Status of a job."""
    job_id: str
    status: str
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None
    output_path: Optional[str] = None


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    engines: List[str]
    version: str


class BookUploadResponse(BaseModel):
    """Response after book upload."""
    book_id: str
    title: str
    chapters: List[str]
    file_path: str


class VoiceLibraryItem(BaseModel):
    """Voice library entry."""
    id: str
    name: str
    file_path: str
    created_at: str
    engine: str = "vibevoice"


class SettingsResponse(BaseModel):
    """Application settings."""
    default_engine: str
    default_voice: str
    default_output_format: str
    use_emotion_tags: bool
    enable_postprocessing: bool


class JobStreamEvent(BaseModel):
    """WebSocket event for job streaming."""
    event_type: str  # "progress", "log", "complete", "error"
    job_id: str
    progress: Optional[float] = None
    message: Optional[str] = None
    timestamp: str



# ============================================================================
# API Application
# ============================================================================

api_app = FastAPI(
    title="Audiobook Creator API",
    description="REST API for audiobook generation with pluggable TTS engines",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@api_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and list available engines."""
    return HealthResponse(
        status="ok",
        engines=list_engines(),
        version="1.0.0"
    )


@api_app.get("/engines", response_model=List[EngineInfo])
async def get_engines():
    """Get all available TTS engines."""
    infos = []
    for name in list_engines():
        engine = get_engine(name)
        if engine:
            info = engine.get_engine_info()
            infos.append(EngineInfo(
                name=info["name"],
                display_name=info["display_name"],
                voice_count=info.get("voice_count", 0),
                capabilities=info.get("capabilities", []),
                min_vram_gb=info.get("min_vram_gb", 0),
                recommended_vram_gb=info.get("recommended_vram_gb", 0)
            ))
    return infos


@api_app.get("/engines/{engine_name}/voices", response_model=List[VoiceInfoResponse])
async def get_voices(engine_name: str):
    """Get available voices for an engine."""
    engine = get_engine(engine_name)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Engine '{engine_name}' not found")
    
    voices = engine.get_available_voices()
    return [
        VoiceInfoResponse(
            id=v.id,
            name=v.name,
            gender=v.gender,
            description=v.description,
            tags=v.tags or []
        )
        for v in voices
    ]


# ============================================================================
# TTS Generation Endpoints
# ============================================================================

@api_app.post("/generate")
async def generate_speech(request: GenerateRequest):
    """
    Generate speech from text.
    
    Returns audio data directly (WAV format).
    """
    try:
        result = await tts_service.generate(
            text=request.text,
            engine=request.engine,
            voice=request.voice,
            emotion=request.emotion,
            speed=request.speed,
            **request.extra_params
        )
        
        # Return audio file
        import io
        from fastapi.responses import StreamingResponse
        
        audio_io = io.BytesIO(result.audio_data)
        return StreamingResponse(
            audio_io,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=generated_audio.wav"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.post("/generate/preview")
async def generate_preview(request: GenerateRequest):
    """
    Generate a short preview (useful for voice testing).
    
    Limits text to first 200 characters.
    """
    # Limit text for preview
    preview_text = request.text[:200]
    if len(request.text) > 200:
        preview_text += "..."
    
    request.text = preview_text
    return await generate_speech(request)


# ============================================================================
# Job Management Endpoints
# ============================================================================

@api_app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """Get all jobs."""
    jobs = job_manager.jobs.values()
    return [
        JobStatus(
            job_id=j.job_id,
            status=j.status.value if hasattr(j.status, 'value') else j.status,
            progress=j.progress,
            message=j.progress_message or "",
            created_at=j.created_at.isoformat() if j.created_at else "",
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            output_path=j.output_path
        )
        for j in jobs
    ]


@api_app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get status of a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return JobStatus(
        job_id=job.job_id,
        status=job.status.value if hasattr(job.status, 'value') else job.status,
        progress=job.progress,
        message=job.progress_message or "",
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        output_path=job.output_path
    )


@api_app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str):
    """Download the output file for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(
            status_code=404, 
            detail="Output file not available (job may still be in progress)"
        )
    
    return FileResponse(
        job.output_path,
        media_type="audio/mp4",
        filename=os.path.basename(job.output_path)
    )


@api_app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job (cancels if running)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    # TODO: Implement job cancellation logic
    job_manager.jobs.pop(job_id, None)
    
    return {"status": "deleted", "job_id": job_id}


# ============================================================================
# Model Management Endpoints
# ============================================================================

@api_app.get("/models", response_model=List[ModelStatus])
async def list_models():
    """List all AI models (TTS, LLM, Transcribe) and their installation status."""
    return model_manager.list_models()


@api_app.post("/models/{model_id}/download")
async def download_model(model_id: str, background_tasks: BackgroundTasks):
    """Start downloading a model in the background."""
    # Check if model exists
    status_list = model_manager.list_models()
    if not any(m.id == model_id for m in status_list):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
    # Check if already installed
    path = model_manager.get_model_path(model_id)
    if path:
        return {"status": "already_installed", "model_id": model_id, "path": path}

    # Define background task
    def _download_task(mid: str):
        try:
            print(f"Starting background download for {mid}...")
            model_manager.download_model(mid)
            print(f"Finished background download for {mid}!")
        except Exception as e:
            print(f"Error downloading {mid}: {e}")

    background_tasks.add_task(_download_task, model_id)
    return {"status": "downloading_started", "model_id": model_id}


@api_app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a downloaded model."""
    # Check if model exists in registry
    status_list = model_manager.list_models()
    if not any(m.id == model_id for m in status_list):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    success = model_manager.delete_model(model_id)
    if not success:
         # If delete failed, it might be because it wasn't installed
         path = model_manager.get_model_path(model_id)
         if not path:
             return {"status": "not_installed", "model_id": model_id}
         
         raise HTTPException(status_code=500, detail="Failed to delete model directory")
         
    return {"status": "deleted", "model_id": model_id}


# ============================================================================
# Book Upload Endpoints
# ============================================================================

@api_app.post("/books/upload", response_model=BookUploadResponse)
async def upload_book(file: UploadFile = File(...)):
    """
    Upload an ebook file for audiobook generation.
    
    Supports: EPUB, MOBI, PDF, TXT
    """
    import uuid
    import shutil
    from datetime import datetime
    
    # Validate file extension
    allowed_extensions = {'.epub', '.mobi', '.pdf', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique book ID
    book_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_dir = os.path.join("uploads", book_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract title from filename
    title = os.path.splitext(file.filename)[0]
    
    # TODO: Extract chapters using calibre
    chapters = ["Chapter 1"]  # Placeholder
    
    return BookUploadResponse(
        book_id=book_id,
        title=title,
        chapters=chapters,
        file_path=file_path
    )


# ============================================================================
# Voice Library Endpoints
# ============================================================================

@api_app.get("/voice-library", response_model=List[VoiceLibraryItem])
async def list_voice_library():
    """Get all custom voices in the voice library."""
    from audiobook.core.voice_manager import voice_manager
    from datetime import datetime
    
    voices = voice_manager.list_voices()
    return [
        VoiceLibraryItem(
            id=v.name,  # Use name as ID
            name=v.name,
            file_path=v.path,
            created_at=datetime.fromtimestamp(os.path.getmtime(v.path)).isoformat() if os.path.exists(v.path) else "",
            engine="vibevoice"
        )
        for v in voices
    ]


@api_app.post("/voice-library", response_model=VoiceLibraryItem)
async def add_voice(name: str, file: UploadFile = File(...)):
    """Add a custom voice to the library."""
    from audiobook.core.voice_manager import voice_manager
    from datetime import datetime
    
    # Read audio data
    audio_data = await file.read()
    
    # Add voice
    voice = voice_manager.add_voice(name, audio_data)
    
    return VoiceLibraryItem(
        id=voice.name,
        name=voice.name,
        file_path=voice.path,
        created_at=datetime.now().isoformat(),
        engine="vibevoice"
    )


@api_app.delete("/voice-library/{voice_id}")
async def delete_voice(voice_id: str):
    """Remove a voice from the library."""
    from audiobook.core.voice_manager import voice_manager
    
    success = voice_manager.delete_voice(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    return {"status": "deleted", "voice_id": voice_id}


# ============================================================================
# Settings Endpoints
# ============================================================================

@api_app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current application settings."""
    return SettingsResponse(
        default_engine=os.environ.get("DEFAULT_TTS_ENGINE", "orpheus"),
        default_voice=os.environ.get("DEFAULT_TTS_VOICE", "zac"),
        default_output_format=os.environ.get("DEFAULT_OUTPUT_FORMAT", "m4b"),
        use_emotion_tags=os.environ.get("USE_EMOTION_TAGS", "true").lower() == "true",
        enable_postprocessing=os.environ.get("ENABLE_POSTPROCESSING", "false").lower() == "true"
    )


@api_app.put("/settings")
async def update_settings(settings: SettingsResponse):
    """Update application settings."""
    # For now, return the settings (full implementation would persist)
    return {"status": "updated", "settings": settings.model_dump()}


# ============================================================================
# WebSocket Job Streaming
# ============================================================================

@api_app.websocket("/jobs/{job_id}/stream")
async def job_stream(websocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.
    
    Events:
    - progress: Job progress percentage
    - log: Log message
    - complete: Job finished successfully
    - error: Job failed
    """
    from starlette.websockets import WebSocket
    from datetime import datetime
    import json
    
    await websocket.accept()
    
    try:
        last_progress = -1
        last_message = ""
        
        while True:
            job = job_manager.get_job(job_id)
            
            if not job:
                await websocket.send_json({
                    "event_type": "error",
                    "job_id": job_id,
                    "message": "Job not found",
                    "timestamp": datetime.now().isoformat()
                })
                break
            
            # Send progress update if changed
            current_progress = job.progress
            current_message = job.progress_message or ""
            
            if current_progress != last_progress or current_message != last_message:
                await websocket.send_json({
                    "event_type": "progress",
                    "job_id": job_id,
                    "progress": current_progress,
                    "message": current_message,
                    "timestamp": datetime.now().isoformat()
                })
                last_progress = current_progress
                last_message = current_message
            
            # Check for completion
            status_value = job.status.value if hasattr(job.status, 'value') else str(job.status)
            if status_value == "completed":
                await websocket.send_json({
                    "event_type": "complete",
                    "job_id": job_id,
                    "progress": 100,
                    "message": "Job completed successfully",
                    "output_path": job.output_path,
                    "timestamp": datetime.now().isoformat()
                })
                break
            elif status_value == "failed":
                await websocket.send_json({
                    "event_type": "error",
                    "job_id": job_id,
                    "message": job.error or "Job failed",
                    "timestamp": datetime.now().isoformat()
                })
                break
            
            # Wait before next poll
            await asyncio.sleep(1)
            
    except Exception as e:
        await websocket.send_json({
            "event_type": "error",
            "job_id": job_id,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })
    finally:
        await websocket.close()



# ============================================================================
# Utility function to mount API with Gradio
# ============================================================================

def mount_api(gradio_app):
    """
    Mount the REST API alongside Gradio.
    
    Usage:
        import gradio as gr
        from audiobook.api import mount_api
        
        demo = gr.Interface(...)
        app = mount_api(demo)
    """
    from fastapi import FastAPI
    from starlette.routing import Mount
    
    # Mount API routes onto Gradio's FastAPI app
    gradio_app.mount("/api", api_app)
    
    return gradio_app
