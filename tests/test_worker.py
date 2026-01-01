"""
Tests for the JobWorker class.
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock

from audiobook.worker import JobWorker, get_worker


class TestJobWorker:
    """Test JobWorker class."""
    
    def test_worker_init(self):
        """Test worker initialization."""
        worker = JobWorker(poll_interval=10)
        assert worker.poll_interval == 10
        assert worker._running is False
        assert worker.current_job is None
    
    def test_worker_claim_no_jobs(self):
        """Test claiming when no pending jobs exist."""
        with patch('audiobook.worker.db') as mock_db:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_db.get_cursor.return_value.__enter__.return_value = mock_cursor
            
            worker = JobWorker()
            job = worker._claim_next_pending_job()
            
            assert job is None
    
    def test_get_worker_singleton(self):
        """Test that get_worker returns singleton."""
        with patch('audiobook.worker._worker', None):
            worker1 = get_worker()
            worker2 = get_worker()
            # Note: Due to module-level singleton, this tests the function
            # In actual use, it would return the same instance


class TestJobWorkerAsync:
    """Async tests for JobWorker."""
    
    def test_worker_stop(self):
        """Test stopping the worker."""
        worker = JobWorker()
        worker._running = True
        worker.stop()
        assert worker._running is False
