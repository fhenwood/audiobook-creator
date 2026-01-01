"""
Integration tests for the Audiobook API.

These tests verify the complete job lifecycle:
1. Create job via API
2. Poll for progress
3. Verify completion/download
4. Resume stalled jobs
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from audiobook.api import api_app


class TestJobAPIIntegration:
    """Integration tests for job submission and lifecycle via API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(api_app)
    
    @pytest.fixture
    def mock_book_file(self, tmp_path):
        """Create a mock book file."""
        book_file = tmp_path / "test_book.txt"
        book_file.write_text("Chapter 1\nThis is a test book.\nWith multiple lines.")
        return str(book_file)
    
    def test_health_endpoint(self, client):
        """Test health check returns available engines."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "engines" in data
        assert isinstance(data["engines"], list)
    
    def test_list_engines(self, client):
        """Test listing available TTS engines."""
        response = client.get("/engines")
        assert response.status_code == 200
        engines = response.json()
        assert isinstance(engines, list)
        # Should have at least orpheus engine
        engine_names = [e["name"] for e in engines]
        assert "orpheus" in engine_names
    
    def test_create_job_missing_book(self, client):
        """Test creating job with missing book file returns 400."""
        response = client.post("/jobs", json={
            "title": "Test Book",
            "book_file_path": "/nonexistent/path.epub",
            "engine": "orpheus",
            "narrator_voice": "zac",
        })
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()
    
    def test_create_job_with_valid_book(self, client, mock_book_file):
        """Test creating job with valid book file."""
        with patch('audiobook.api.job_service') as mock_service:
            # Mock the job creation
            mock_job = MagicMock()
            mock_job.job_id = "test-job-123"
            mock_job.status.value = "pending"
            mock_job.percent_complete = 0
            mock_job.progress = "Queued"
            mock_job.created_at = "2024-01-01T00:00:00Z"
            mock_job.output_file = None
            
            mock_service.create_and_run_job = AsyncMock(return_value="test-job-123")
            
            with patch('audiobook.api.job_manager') as mock_jm:
                mock_jm.create_job.return_value = mock_job
                mock_jm.get_job.return_value = mock_job
                mock_jm.get_job_converted_book_path.return_value = "/tmp/test.txt"
                
                with patch('audiobook.api.background_runner') as mock_runner:
                    mock_runner.submit_job.return_value = True
                    
                    response = client.post("/jobs", json={
                        "title": "Test Book",
                        "book_file_path": mock_book_file,
                        "engine": "orpheus",
                        "narrator_voice": "zac",
                        "output_format": "m4b",
                    })
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["job_id"] == "test-job-123"
                    assert data["status"] == "pending"

    def test_get_job_status(self, client):
        """Test getting job status by ID."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_job = MagicMock()
            mock_job.job_id = "test-job-456"
            mock_job.status.value = "in_progress"
            mock_job.percent_complete = 50
            mock_job.progress = "Processing line 500 of 1000"
            mock_job.created_at = datetime.now()
            mock_job.completed_at = None
            mock_job.output_file = None
            mock_jm.get_job.return_value = mock_job
            
            response = client.get("/jobs/test-job-456")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-456"
            assert data["status"] == "in_progress"
            assert data["progress"] == 50
    
    def test_get_nonexistent_job(self, client):
        """Test getting nonexistent job returns 404."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_jm.get_job.return_value = None
            
            response = client.get("/jobs/nonexistent-job")
            assert response.status_code == 404

    def test_resume_stalled_job(self, client):
        """Test resuming a stalled job."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_job = MagicMock()
            mock_job.job_id = "stalled-job-789"
            mock_job.status.value = "stalled"
            mock_job.percent_complete = 75
            mock_job.progress = "Resuming..."
            mock_job.created_at = datetime.now()
            mock_job.get_checkpoint.return_value = {"line_number": 750}
            mock_jm.get_job.return_value = mock_job
            
            with patch('audiobook.api.background_runner') as mock_runner:
                mock_runner.submit_job.return_value = True
                
                response = client.post("/jobs/stalled-job-789/resume")
                assert response.status_code == 200
                data = response.json()
                assert data["job_id"] == "stalled-job-789"
                assert data["status"] == "running"

    def test_resume_non_stalled_job_fails(self, client):
        """Test that resuming a non-stalled job returns error."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_job = MagicMock()
            mock_job.job_id = "running-job"
            mock_job.status.value = "in_progress"
            mock_jm.get_job.return_value = mock_job
            
            response = client.post("/jobs/running-job/resume")
            assert response.status_code == 400
            assert "not stalled" in response.json()["detail"].lower()

    def test_delete_job(self, client):
        """Test deleting a job."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_jm.delete_job.return_value = True
            
            response = client.delete("/jobs/job-to-delete")
            assert response.status_code == 200

    def test_list_jobs(self, client):
        """Test listing all jobs."""
        with patch('audiobook.api.job_manager') as mock_jm:
            mock_job1 = MagicMock()
            mock_job1.job_id = "job-1"
            mock_job1.book_title = "Book 1"
            mock_job1.status.value = "completed"
            mock_job1.percent_complete = 100
            mock_job1.created_at = "2024-01-01T00:00:00Z"
            mock_job1.completed_at = "2024-01-01T01:00:00Z"
            mock_job1.output_file = "/path/to/book1.m4b"
            
            mock_jm.get_all_jobs.return_value = [mock_job1]
            
            response = client.get("/jobs")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
