"""
TTS Engines Package

This package contains the pluggable TTS engine system.
"""

from audiobook.tts.engines.base import (
    TTSEngine,
    VoiceInfo,
    GenerationResult,
    EngineConfig,
    EngineCapability
)
from audiobook.tts.engines.registry import (
    engine_registry,
    register_engine,
    get_engine,
    list_engines
)

__all__ = [
    # Base classes
    "TTSEngine",
    "VoiceInfo", 
    "GenerationResult",
    "EngineConfig",
    "EngineCapability",
    # Registry
    "engine_registry",
    "register_engine",
    "get_engine",
    "list_engines",
]

# Auto-register built-in engines when package is imported
def _auto_register_engines():
    """Automatically register all built-in engines."""
    import importlib
    import os
    
    engines_dir = os.path.dirname(__file__)
    skip_files = {"__init__.py", "base.py", "registry.py"}
    
    for filename in os.listdir(engines_dir):
        if filename.endswith(".py") and filename not in skip_files:
            module_name = filename[:-3]
            try:
                importlib.import_module(f".{module_name}", package=__name__)
            except Exception as e:
                print(f"⚠️ Failed to load engine module {module_name}: {e}")

# Run auto-registration
_auto_register_engines()
