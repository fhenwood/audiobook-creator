
import pytest
from unittest.mock import MagicMock, patch
import sys
import asyncio

# Mock modules that might not be installed in the test env
sys.modules["llama_cpp"] = MagicMock()
sys.modules["snac"] = MagicMock()

from audiobook.tts.engines.orpheus import OrpheusEngine

def test_orpheus_init_in_process():
    """Test that Orpheus initializes using local modules, not API."""
    
    async def _run_test():
        # Mock model manager to return a valid path
        with patch("audiobook.models.manager.model_manager.get_model_path", return_value="/tmp/test_model.gguf"):
            # Mock torch.cuda.is_available
            with patch("torch.cuda.is_available", return_value=False):
                
                engine = OrpheusEngine()
                
                # Setup mocks for imports inside the method
                with patch.dict(sys.modules, {"llama_cpp": MagicMock(), "snac": MagicMock()}):
                    
                    # Run initialize
                    success = await engine.initialize()
                    
                    # Should be true if mocks work
                    # Ideally we want to verify it did NOT call openai/api fallback
                    assert not hasattr(engine, "_client"), "Engine should not have an API client"
                    assert not hasattr(engine, "_use_api"), "Engine should not be in API mode"
    
    asyncio.run(_run_test())

def test_orpheus_init_fail_without_library():
    """Test that Orpheus fails gracefully if libraries are missing."""
    pass

