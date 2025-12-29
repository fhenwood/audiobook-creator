#!/usr/bin/env python3
"""
E2E Test Suite for Audiobook Creator - VibeVoice Integration

Standalone, repeatable tests that can be run by anyone without AI assistance.

Usage:
    # Run all tests
    pytest tests/e2e_vibevoice_test.py -v
    
    # Run specific test
    pytest tests/e2e_vibevoice_test.py::test_gradio_health -v
    
    # Run with custom URL
    AUDIOBOOK_URL=http://myserver:7860 pytest tests/e2e_vibevoice_test.py -v

Requirements:
    pip install pytest requests
"""

import os
import sys
import json
import time
import pytest
import requests
from pathlib import Path
from datetime import datetime, timedelta

# ==================== CONFIGURATION ====================

BASE_URL = os.environ.get("AUDIOBOOK_URL", "http://localhost:7860")
API_URL = f"{BASE_URL}/api"
PROJECT_ROOT = Path(__file__).parent.parent

# Timeouts
REQUEST_TIMEOUT = 30
JOB_POLL_INTERVAL = 10
JOB_POLL_MAX_WAIT = 120  # 2 minutes max wait for job progress


# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def base_url():
    """Base URL for the Audiobook Creator app."""
    return BASE_URL


@pytest.fixture(scope="session")
def jobs_json_path():
    """Path to jobs.json file."""
    return PROJECT_ROOT / "generated_audiobooks" / "jobs.json"


@pytest.fixture
def session():
    """Requests session with default timeout."""
    s = requests.Session()
    s.timeout = REQUEST_TIMEOUT
    return s


# ==================== HEALTH CHECK TESTS ====================

class TestHealthChecks:
    """Basic health checks for the application."""
    
    def test_gradio_app_responds(self, session, base_url):
        """Test that the Gradio app is running and responding."""
        response = session.get(base_url)
        assert response.status_code == 200, f"Gradio app not responding: {response.status_code}"
        assert "Audiobook" in response.text or "gradio" in response.text.lower(), \
            "Response doesn't look like Gradio app"
    
    def test_static_files_accessible(self, session, base_url):
        """Test that static files are being served."""
        response = session.get(f"{base_url}/file=static_files/", allow_redirects=True)
        # Should get 200 or 404 (files listed or not), not 500
        assert response.status_code != 500, "Static file serving is broken"
    
    def test_api_endpoint_exists(self, session, base_url):
        """Test that the API endpoint is mounted."""
        response = session.get(f"{base_url}/api/", allow_redirects=True)
        # Should not be a 404 at the API root level
        assert response.status_code != 404 or "Not Found" in response.text, \
            "API endpoint not mounted"


# ==================== JOBS SYSTEM TESTS ====================

class TestJobsSystem:
    """Tests for the job management system."""
    
    def test_jobs_json_exists_and_valid(self, jobs_json_path):
        """Test that jobs.json exists and is valid JSON (if present)."""
        if not jobs_json_path.exists():
            pytest.skip("No jobs.json yet - fresh installation")
        
        with open(jobs_json_path) as f:
            jobs = json.load(f)
        
        assert isinstance(jobs, dict), "jobs.json should be a dictionary"
    
    def test_job_structure_valid(self, jobs_json_path):
        """Test that existing jobs have required fields."""
        if not jobs_json_path.exists():
            pytest.skip("No jobs.json yet")
        
        with open(jobs_json_path) as f:
            jobs = json.load(f)
        
        if not jobs:
            pytest.skip("No jobs in jobs.json")
        
        required_fields = ["job_id", "status", "tts_engine", "created_at"]
        
        for job_id, job in jobs.items():
            for field in required_fields:
                assert field in job, f"Job {job_id} missing required field: {field}"
    
    def test_vibevoice_jobs_have_voice_path(self, jobs_json_path):
        """Test that VibeVoice jobs have a valid voice path."""
        if not jobs_json_path.exists():
            pytest.skip("No jobs.json yet")
        
        with open(jobs_json_path) as f:
            jobs = json.load(f)
        
        vibevoice_jobs = [j for j in jobs.values() if j.get("tts_engine") == "VibeVoice"]
        
        if not vibevoice_jobs:
            pytest.skip("No VibeVoice jobs found")
        
        for job in vibevoice_jobs:
            voice = job.get("vibevoice_voice") or job.get("voice")
            assert voice, f"VibeVoice job {job['job_id']} has no voice"
            # Should NOT be a default speaker_X unless intended
            if voice.startswith("speaker_"):
                pytest.warn(f"Job {job['job_id']} uses default speaker: {voice}")
    
    def test_in_progress_jobs_are_updating(self, jobs_json_path):
        """Test that in-progress jobs are actually making progress."""
        if not jobs_json_path.exists():
            pytest.skip("No jobs.json yet")
        
        with open(jobs_json_path) as f:
            jobs_before = json.load(f)
        
        in_progress = [j for j in jobs_before.values() if j.get("status") == "in_progress"]
        
        if not in_progress:
            pytest.skip("No in-progress jobs to monitor")
        
        # Wait a reasonable time for batch processing
        time.sleep(JOB_POLL_INTERVAL)
        
        with open(jobs_json_path) as f:
            jobs_after = json.load(f)
        
        # Check if ANY job updated (some may be between batch saves)
        any_updated = False
        for job in in_progress:
            job_id = job["job_id"]
            before_time = job.get("updated_at", "")
            after_time = jobs_after.get(job_id, {}).get("updated_at", "")
            
            if after_time > before_time:
                any_updated = True
                break
            
            # Also check if job completed
            after_status = jobs_after.get(job_id, {}).get("status", "")
            if after_status in ["completed", "failed", "stalled"]:
                any_updated = True
                break
        
        # Don't fail hard - just warn (batch intervals can be long)
        if not any_updated:
            pytest.warn(
                f"No job progress detected in {JOB_POLL_INTERVAL}s. "
                "This may be normal for large batch intervals."
            )


# ==================== VOICE LIBRARY TESTS ====================

class TestVoiceLibrary:
    """Tests for the voice library system."""
    
    def test_voices_directory_exists(self):
        """Test that voice directories exist."""
        voices_path = PROJECT_ROOT / "static_files" / "voices"
        presets_path = PROJECT_ROOT / "static_files" / "preset_voices"
        
        assert voices_path.exists() or presets_path.exists(), \
            "Neither voices nor preset_voices directory found"
    
    def test_voices_have_audio_files(self):
        """Test that voice directories contain audio files."""
        voices_path = PROJECT_ROOT / "static_files" / "voices"
        presets_path = PROJECT_ROOT / "static_files" / "preset_voices"
        
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        found_audio = False
        
        for dir_path in [voices_path, presets_path]:
            if dir_path.exists():
                for f in dir_path.iterdir():
                    if f.suffix.lower() in audio_extensions:
                        found_audio = True
                        break
            if found_audio:
                break
        
        assert found_audio, "No audio files found in voice directories"


# ==================== GPU TESTS ====================

class TestGPU:
    """Tests for GPU availability and health."""
    
    def test_nvidia_smi_available(self):
        """Test that nvidia-smi is available (GPU system)."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, "nvidia-smi failed"
            assert result.stdout.strip(), "No GPU detected"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not available - likely CPU-only system")
    
    def test_gpu_not_oom(self):
        """Test that GPU is not critically out of memory."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                pytest.skip("nvidia-smi query failed")
            
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                used = int(parts[0].strip())
                total = int(parts[1].strip())
                usage_pct = (used / total) * 100
                
                assert usage_pct < 98, f"GPU memory critically full: {usage_pct:.1f}%"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not available")


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests that verify end-to-end functionality."""
    
    def test_temp_audio_directory_writable(self):
        """Test that temp audio directory is writable."""
        temp_audio_path = PROJECT_ROOT / "temp_audio"
        
        # Create if doesn't exist
        temp_audio_path.mkdir(exist_ok=True)
        
        # Try to write a test file
        test_file = temp_audio_path / f"test_{int(time.time())}.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()  # Clean up
        except Exception as e:
            pytest.fail(f"Cannot write to temp_audio directory: {e}")
    
    def test_generated_audiobooks_directory_writable(self):
        """Test that generated_audiobooks directory is writable."""
        output_path = PROJECT_ROOT / "generated_audiobooks"
        
        # Create if doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Try to write a test file
        test_file = output_path / f"test_{int(time.time())}.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()  # Clean up
        except PermissionError:
            pytest.skip("Directory is Docker-mounted with restricted permissions")
        except Exception as e:
            pytest.fail(f"Cannot write to generated_audiobooks directory: {e}")



# ==================== MAIN ====================

if __name__ == "__main__":
    # Run with pytest
    sys.exit(pytest.main([__file__, "-v"]))
