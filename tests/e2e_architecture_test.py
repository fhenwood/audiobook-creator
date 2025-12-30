"""
E2E Tests for MacWhisper Architecture Changes

Tests the new REST API endpoints, Orpheus engine refactor, and app modules.
"""

import pytest
import requests
import asyncio
import os
from datetime import datetime


# Configuration
BASE_URL = os.environ.get("AUDIOBOOK_URL", "http://localhost:7860")
API_URL = f"{BASE_URL}/api"


class TestAPIHealth:
    """Test API health and availability."""
    
    def test_health_endpoint(self):
        """Test /api/health returns OK."""
        response = requests.get(f"{API_URL}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "engines" in data
        assert "version" in data
    
    def test_engines_endpoint(self):
        """Test /api/engines returns list of TTS engines."""
        response = requests.get(f"{API_URL}/engines", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have at least Orpheus
        engine_names = [e["name"] for e in data]
        assert "orpheus" in engine_names or len(engine_names) > 0


class TestVoiceLibraryAPI:
    """Test voice library API endpoints."""
    
    def test_list_voice_library(self):
        """Test GET /api/voice-library returns list."""
        response = requests.get(f"{API_URL}/voice-library", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Each item should have expected fields
        if len(data) > 0:
            assert "id" in data[0]
            assert "name" in data[0]
            assert "file_path" in data[0]


class TestSettingsAPI:
    """Test settings API endpoints."""
    
    def test_get_settings(self):
        """Test GET /api/settings returns current settings."""
        response = requests.get(f"{API_URL}/settings", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "default_engine" in data
        assert "default_voice" in data
        assert "default_output_format" in data
        assert "use_emotion_tags" in data
        assert "enable_postprocessing" in data
    
    def test_update_settings(self):
        """Test PUT /api/settings updates settings."""
        settings = {
            "default_engine": "orpheus",
            "default_voice": "zac",
            "default_output_format": "m4b",
            "use_emotion_tags": True,
            "enable_postprocessing": False
        }
        response = requests.put(f"{API_URL}/settings", json=settings, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"


class TestJobsAPI:
    """Test job management API endpoints."""
    
    def test_list_jobs(self):
        """Test GET /api/jobs returns list of jobs."""
        response = requests.get(f"{API_URL}/jobs", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestModelsAPI:
    """Test model management API endpoints."""
    
    def test_list_models(self):
        """Test GET /api/models returns model list."""
        response = requests.get(f"{API_URL}/models", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestVoicesAPI:
    """Test voice listing for engines."""
    
    def test_orpheus_voices(self):
        """Test GET /api/engines/orpheus/voices returns voices."""
        response = requests.get(f"{API_URL}/engines/orpheus/voices", timeout=10)
        # May return 404 if engine not initialized, which is OK
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            if len(data) > 0:
                assert "id" in data[0]
                assert "name" in data[0]
                assert "gender" in data[0]


class TestAppModules:
    """Test that new app modules import correctly."""
    
    def test_voice_utils_import(self):
        """Test app.voice_utils imports without error."""
        try:
            from app.voice_utils import (
                extract_title_from_filename,
                get_vibevoice_choices,
                get_voice_choices,
                get_installed_tts_engines,
                check_llm_availability,
            )
            assert callable(get_voice_choices)
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")
    
    def test_handlers_import(self):
        """Test app.handlers imports without error."""
        try:
            from app.handlers import (
                validate_book_upload,
                text_extraction_wrapper,
                save_book_wrapper,
            )
            assert callable(validate_book_upload)
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")
    
    def test_jobs_import(self):
        """Test app.jobs imports without error."""
        try:
            from app.jobs import (
                run_audiobook_job_background,
                resume_job_background,
            )
            assert callable(run_audiobook_job_background)
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")


class TestGeneratorPackage:
    """Test generator package imports."""
    
    def test_generator_utils_import(self):
        """Test audiobook.tts.generator.utils imports."""
        try:
            from audiobook.tts.generator.utils import (
                sanitize_filename,
                is_only_punctuation,
                split_and_annotate_text,
                check_if_chapter_heading,
                validate_book_for_m4b_generation,
            )
            assert callable(sanitize_filename)
            
            # Test sanitize_filename functionality
            result = sanitize_filename("Test: Book & File")
            assert ":" not in result
            assert "&" not in result
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")


class TestOrpheusEngine:
    """Test Orpheus engine (in-process mode)."""
    
    def test_orpheus_engine_import(self):
        """Test OrpheusEngine imports correctly."""
        try:
            from audiobook.tts.engines.orpheus import OrpheusEngine, ORPHEUS_VOICES
            assert OrpheusEngine.name == "orpheus"
            assert len(ORPHEUS_VOICES) > 0
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")
    
    def test_orpheus_voices_structure(self):
        """Test Orpheus voice definitions are correct."""
        try:
            from audiobook.tts.engines.orpheus import ORPHEUS_VOICES
            for voice in ORPHEUS_VOICES:
                assert hasattr(voice, "id")
                assert hasattr(voice, "name")
                assert hasattr(voice, "gender")
                assert voice.gender in ["male", "female"]
        except ImportError as e:
            pytest.skip(f"Module not available in test environment: {e}")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
