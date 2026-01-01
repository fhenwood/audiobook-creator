"""
Model Download Manager

Manages the registry, download, and deletion of AI models (TTS, LLM, Transcribe).
Centralizes model management to provide a unified interface for the frontend.
"""

import os
import shutil
import logging
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel
from huggingface_hub import snapshot_download, scan_cache_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from audiobook.config import settings


class ModelType(str, Enum):
    TTS = "tts"
    LLM = "llm"
    TRANSCRIBE = "transcribe"
    VOICE_CONVERSION = "vc"
    VOICE_ANALYZER = "voice-analyzer"

class ModelDefinition(BaseModel):
    """Static definition of a supported model."""
    id: str
    name: str
    type: ModelType
    description: str
    source_repo: str  # HuggingFace repo ID
    files: Optional[List[str]] = None  # Specific files to download (if None, download all)
    default_filename: Optional[str] = None # For single-file models (GGUF)
    size_estimate_gb: float
    required_ram_gb: float

class ModelStatus(BaseModel):
    """Runtime status of a model."""
    id: str
    installed: bool
    downloaded_size_gb: float = 0.0
    path: Optional[str] = None
    definition: ModelDefinition

# ============================================================================
# Supported Models Registry
# ============================================================================

SUPPORTED_MODELS: Dict[str, ModelDefinition] = {
    # TTS Models
    "maya-1": ModelDefinition(
        id="maya-1",
        name="Maya1",
        type=ModelType.TTS,
        description="State-of-the-art expressive TTS (3B parameters)",
        source_repo="maya-research/Maya1",
        size_estimate_gb=7.0,
        required_ram_gb=8.0
    ),
    "voice-analyzer-qwen": ModelDefinition(
        id="voice-analyzer-qwen",
        name="Qwen2-Audio 7B",
        type=ModelType.VOICE_ANALYZER,
        description="Voice Analysis and Description (7B parameters)",
        source_repo="Qwen/Qwen2-Audio-7B-Instruct",
        size_estimate_gb=15.0,
        required_ram_gb=16.0
    ),
    "vibevoice-7b": ModelDefinition(
        id="vibevoice-7b",
        name="VibeVoice 7B",
        type=ModelType.TTS,
        description="High-fidelity multi-speaker TTS (7B parameters)",
        source_repo="vibevoice/VibeVoice-7B",
        size_estimate_gb=15.0,
        required_ram_gb=24.0
    ),
    "vibevoice-realtimemma-500m": ModelDefinition(
        id="vibevoice-realtimemma-500m",
        name="VibeVoice 500M (Realtime)",
        type=ModelType.TTS,
        description="Faster, lighter VibeVoice model for real-time applications",
        source_repo="vibevoice/VibeVoice-Realtime-0.5B",
        size_estimate_gb=2.0,
        required_ram_gb=4.0
    ),
    "kokoro-82m": ModelDefinition(
        id="kokoro-82m",
        name="Kokoro 82M",
        type=ModelType.TTS,
        description="Lightweight, high-quality TTS model (82M parameters)",
        source_repo="hexgrad/Kokoro-82M",
        size_estimate_gb=0.5,
        required_ram_gb=2.0
    ),
    "xtts-v2": ModelDefinition(
        id="xtts-v2",
        name="XTTS v2",
        type=ModelType.TTS,
        description="Coqui XTTS v2 - Good quality, fast, multilingual",
        source_repo="coqui/XTTS-v2",
        size_estimate_gb=3.0,
        required_ram_gb=6.0
    ),
    
    "orpheus-3b-0.1-ft-Q8_0.gguf": ModelDefinition(
        id="orpheus-3b-0.1-ft-Q8_0.gguf",
        name="Orpheus 3B (Q8_0)",
        type=ModelType.TTS,
        description="Orpheus 3B TTS Model (Q8_0 GGUF) - High Quality",
        source_repo="unsloth/orpheus-3b-0.1-ft-GGUF",
        files=["orpheus-3b-0.1-ft-Q8_0.gguf"],
        default_filename="orpheus-3b-0.1-ft-Q8_0.gguf",
        size_estimate_gb=3.5,
        required_ram_gb=4.0
    ),

    "orpheus-3b-0.1-ft-Q4_K_M.gguf": ModelDefinition(
        id="orpheus-3b-0.1-ft-Q4_K_M.gguf",
        name="Orpheus 3B (Q4_K_M)",
        type=ModelType.TTS,
        description="Orpheus 3B TTS Model (Q4 GGUF) - Efficient",
        source_repo="unsloth/orpheus-3b-0.1-ft-GGUF",
        files=["orpheus-3b-0.1-ft-Q4_K_M.gguf"],
        default_filename="orpheus-3b-0.1-ft-Q4_K_M.gguf",
        size_estimate_gb=2.1,
        required_ram_gb=3.0
    ),

    # LLM Models (GGUF for efficiency)
    "gpt-oss-20b-gguf": ModelDefinition(
        id="gpt-oss-20b-gguf",
        name="GPT-OSS 20B (GGUF)",
        type=ModelType.LLM,
        description="Powerful 20B parameter model for high-quality text analysis",
        source_repo="unsloth/gpt-oss-20b-GGUF",
        files=["gpt-oss-20b-Q4_K_M.gguf"],
        default_filename="gpt-oss-20b-Q4_K_M.gguf",
        size_estimate_gb=12.0,
        required_ram_gb=16.0
    ),
    "llama-3-8b-instruct-gguf": ModelDefinition(

        id="llama-3-8b-instruct-gguf",
        name="Llama 3 8B Instruct (Q4_K_M)",
        type=ModelType.LLM,
        description="Intelligent instruction-following model for text analysis and emotion tagging",
        source_repo="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        files=["Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"],
        default_filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        size_estimate_gb=4.9,
        required_ram_gb=8.0
    ),
    "mistral-7b-instruct-v0.3-gguf": ModelDefinition(
        id="mistral-7b-instruct-v0.3-gguf",
        name="Mistral 7B Instruct v0.3 (Q4_K_M)",
        type=ModelType.LLM,
        description="Versatile LLM, good alternative to Llama 3",
        source_repo="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
        files=["Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"],
        default_filename="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
        size_estimate_gb=4.4,
        required_ram_gb=8.0
    ),
    
    # Transcription Models
    "whisper-large-v3": ModelDefinition(
        id="whisper-large-v3",
        name="Whisper Large v3",
        type=ModelType.TRANSCRIBE,
        description="State-of-the-art speech recognition and transcription",
        source_repo="openai/whisper-large-v3",
        size_estimate_gb=3.0,
        required_ram_gb=4.0
    ),
    "whisper-medium": ModelDefinition(
        id="whisper-medium",
        name="Whisper Medium",
        type=ModelType.TRANSCRIBE,
        description="Balanced speed and accuracy for transcription",
        source_repo="openai/whisper-medium",
        size_estimate_gb=1.5,
        required_ram_gb=2.0
    ),
}

class ModelManager:
    def __init__(self):
        self.base_dir = settings.models_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def list_models(self) -> List[ModelStatus]:
        """List all supported models and their installation status."""
        statuses = []
        for model_id, definition in SUPPORTED_MODELS.items():
            installed, size, path = self._check_installation(definition)
            statuses.append(ModelStatus(
                id=model_id,
                installed=installed,
                downloaded_size_gb=size,
                path=path,
                definition=definition
            ))
        return statuses
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get the path to a model if installed."""
        if model_id not in SUPPORTED_MODELS:
            return None
        
        definition = SUPPORTED_MODELS[model_id]
        installed, _, path = self._check_installation(definition)
        return path if installed else None

    def download_model(self, model_id: str) -> str:
        """
        Download a model from HuggingFace.
        Returns the path where it was downloaded.
        """
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        definition = SUPPORTED_MODELS[model_id]
        target_dir = os.path.join(self.base_dir, definition.type.value, model_id)
        os.makedirs(target_dir, exist_ok=True)
        
        logger.info(f"Downloading {model_id} from {definition.source_repo}...")
        
        try:
            # Download using huggingface_hub
            # We download to a local directory instead of cache for easier management
            path = snapshot_download(
                repo_id=definition.source_repo,
                revision="main",
                local_dir=target_dir,
                local_dir_use_symlinks=False, # Copy real files
                allow_patterns=definition.files if definition.files else None
            )
            logger.info(f"Successfully downloaded {model_id} to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            # Cleanup partial download if needed
            raise RuntimeError(f"Download failed: {e}") from e

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
            
        definition = SUPPORTED_MODELS[model_id]
        target_dir = os.path.join(self.base_dir, definition.type.value, model_id)
        
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                logger.info(f"Deleted model {model_id} at {target_dir}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete {model_id}: {e}")
                return False
        return False

    def _check_installation(self, definition: ModelDefinition):
        """Check if model is installed and return (installed, size_gb, path)."""
        target_dir = os.path.join(self.base_dir, definition.type.value, definition.id)
        
        if not os.path.exists(target_dir):
            return False, 0.0, None
        
        # Check if empty (or only boilerplate)
        if not os.listdir(target_dir):
            return False, 0.0, None
            
        # Calculate size
        total_size = 0
        for dirpath, _, filenames in os.walk(target_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        
        size_gb = total_size / (1024**3)
        
        # Check for weight files to confirm actual installation
        valid_extensions = {'.bin', '.safetensors', '.pt', '.pth', '.ckpt', '.gguf', '.onnx', '.msgpack'}
        has_weights = False
        for dirpath, _, filenames in os.walk(target_dir):
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext.lower() in valid_extensions and os.path.getsize(os.path.join(dirpath, f)) > 10 * 1024 * 1024:
                    has_weights = True
                    break
            if has_weights:
                break
        
        installed = has_weights and size_gb > 0.01
        
        return installed, round(size_gb, 2), target_dir

# Singleton instance
model_manager = ModelManager()
