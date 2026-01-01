
import pytest
import os
import json
import sqlite3
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from audiobook.utils.job_manager import JobManager, Job, JobStatus, JobCheckpoint
from audiobook.utils.database import Database

# Mock the settings to avoid loading actual environment
@pytest.fixture(autouse=True)
def mock_settings():
    with patch('audiobook.config.settings') as mock:
        yield mock

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    print(f"DEBUG: Created temp_dir {temp_dir}")
    yield temp_dir
    shutil.rmtree(temp_dir)
    print(f"DEBUG: Removed temp_dir {temp_dir}")

@pytest.fixture
def test_db(temp_dir):
    """Initialize a temporary database for testing."""
    db_path = os.path.join(temp_dir, "test_jobs.db")
    print(f"DEBUG: using db_path {db_path}")
    
    # Patch the global database instance to use our temp DB
    with patch('audiobook.utils.job_manager.db') as mock_db:
        # Setup the real Database behavior but pointing to temp file
        real_db = Database()
        real_db.DB_FILE = db_path
        real_db._ensure_db_directory()
        real_db._init_schema()
        
        # Proxy calls to the real DB instance
        mock_db.get_cursor = real_db.get_cursor
        
        yield mock_db
        
        real_db.close()

@pytest.fixture
def job_manager(test_db, temp_dir):
    """Initialize JobManager with a temporary directory."""
    # Reset the singleton instance
    JobManager._instance = None
    
    # Preventing migration from running with real file
    with patch('audiobook.utils.job_manager.JobManager._migrate_json_to_db') as mock_migrate:
        with patch('audiobook.utils.job_manager.JobManager.TEMP_AUDIO_BASE', new=os.path.join(temp_dir, "temp_audio")):
            manager = JobManager()
            # Ensure it uses temp jobs file (though migration is mocked now)
            manager.JOBS_FILE = os.path.join(temp_dir, "jobs.json")
            
            # Helper to access the global job_manager used by AutoResumeService
            with patch('audiobook.utils.job_manager.job_manager', manager):
                yield manager

class TestJobManager:

    def test_create_job(self, job_manager):
        """Test creating a new job."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        assert job.job_id is not None
        assert job.book_title == "Test Book"
        assert job.status == JobStatus.PENDING.value
        assert job.percent_complete == 0.0
        
        # Verify it exists in DB
        fetched_job = job_manager.get_job(job.job_id)
        assert fetched_job is not None
        assert fetched_job.job_id == job.job_id

    def test_job_directory_creation(self, job_manager):
        """Test that job directories are created."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        job_dir = job_manager.get_job_dir(job.job_id)
        line_segments_dir = job_manager.get_job_line_segments_dir(job.job_id)
        
        assert os.path.exists(job_dir)
        assert os.path.exists(line_segments_dir)

    def test_update_job_progress(self, job_manager):
        """Test updating job progress."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        job_manager.update_job_progress(job.job_id, "Processing...", 50.0)
        
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.status == JobStatus.IN_PROGRESS.value
        assert updated_job.progress == "Processing..."
        assert updated_job.percent_complete == 50.0

    def test_job_checkpoint(self, job_manager):
        """Test saving and retrieving job checkpoints."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        checkpoint = JobCheckpoint(
            total_lines=100,
            lines_completed=25,
            book_file_path="/path/to/book.epub"
        )
        
        job_manager.update_job_checkpoint(job.job_id, checkpoint)
        
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.checkpoint is not None
        
        fetched_checkpoint = updated_job.get_checkpoint()
        assert fetched_checkpoint.total_lines == 100
        assert fetched_checkpoint.lines_completed == 25

    def test_complete_job(self, job_manager):
        """Test marking a job as complete."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        output_file = "/path/to/output.m4b"
        job_manager.complete_job(job.job_id, output_file)
        
        completed_job = job_manager.get_job(job.job_id)
        assert completed_job.status == JobStatus.COMPLETED.value
        assert completed_job.percent_complete == 100.0
        assert completed_job.output_file == output_file

    def test_fail_job(self, job_manager):
        """Test marking a job as failed."""
        job = job_manager.create_job(
            book_title="Test Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        error_msg = "Something went wrong"
        job_manager.fail_job(job.job_id, error_msg)
        
        failed_job = job_manager.get_job(job.job_id)
        assert failed_job.status == JobStatus.FAILED.value
        assert failed_job.error_message == error_msg
        assert "Failed" in failed_job.progress

    def test_mark_line_complete(self, job_manager):
        """Test marking individual lines as complete in DB."""
        job = job_manager.create_job(
            book_title="Line Track Test",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        # Initially no lines completed
        completed = job_manager.get_completed_lines(job.job_id)
        assert len(completed) == 0
        
        # Mark some lines complete
        job_manager.mark_line_complete(job.job_id, 0)
        job_manager.mark_line_complete(job.job_id, 5)
        job_manager.mark_line_complete(job.job_id, 10)
        
        completed = job_manager.get_completed_lines(job.job_id)
        assert completed == {0, 5, 10}
        
        # Mark same line again (should not duplicate)
        job_manager.mark_line_complete(job.job_id, 5)
        completed = job_manager.get_completed_lines(job.job_id)
        assert completed == {0, 5, 10}

    def test_stalled_job_detection(self, job_manager, test_db):
        """Test detecting stalled jobs."""
        job = job_manager.create_job(
            book_title="Stalled Book",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        # Simulate progress with checkpoint
        job_manager.update_job_progress(job.job_id, "Working...", 10.0)
        
        # Add checkpoint (required for resumability/stalling)
        checkpoint = JobCheckpoint(total_lines=100, lines_completed=10)
        job_manager.update_job_checkpoint(job.job_id, checkpoint)
        
        # Manually backdate the last_activity in DB
        old_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        
        # Use the test_db fixture directly
        with test_db.get_cursor() as cursor:
            # Update the JSON blob AND the updated_at column
            # First fetch to get JSON
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job.job_id,))
            row = cursor.fetchone()
            data = json.loads(row['data'])
            data['last_activity'] = old_time
            
            cursor.execute(
                "UPDATE jobs SET updated_at = ?, data = ? WHERE id = ?",
                (old_time, json.dumps(data), job.job_id)
            )
            
        # Check for stalled jobs (timeout 5 mins)
        stalled = job_manager.check_for_stalled_jobs(stall_timeout_seconds=300)
        
        assert job.job_id in stalled
        
        stalled_job = job_manager.get_job(job.job_id)
        assert stalled_job.status == JobStatus.STALLED.value

    def test_delete_job(self, job_manager):
        """Test deleting a job."""
        job = job_manager.create_job(
            book_title="Delete Me",
            tts_engine="Orpheus",
            voice="zac",
            output_format="M4B"
        )
        
        job_id = job.job_id
        assert job_manager.get_job(job_id) is not None
        
        job_manager.delete_job(job_id)
        assert job_manager.get_job(job_id) is None
        
        # Verify directory is gone
        assert not os.path.exists(job_manager.get_job_dir(job_id))

class TestAutoResumeService:
    @pytest.fixture
    def auto_resume_service(self):
        """Get the singleton instance."""
        from audiobook.utils.job_manager import AutoResumeService
        # Reset singleton and state
        AutoResumeService._instance = None
        service = AutoResumeService()
        yield service
        service.stop()

    def test_auto_resume_trigger(self, auto_resume_service, job_manager, test_db):
        """Test that auto-resume calls the callback."""
        # Force clean DB
        with test_db.get_cursor() as cursor:
            cursor.execute("DELETE FROM jobs")
            
        # Create a stalled job
        job = job_manager.create_job("Stalled Job", "Orpheus", "zac", "M4B")
        checkpoint = JobCheckpoint(total_lines=100, lines_completed=10)
        job_manager.update_job_checkpoint(job.job_id, checkpoint)
        
        # Verify checkpoint is set (this fetches fresh job from DB)
        job = job_manager.get_job(job.job_id)
        assert job.checkpoint is not None
        
        # Mark as stalled manually
        job.status = JobStatus.STALLED.value
        with test_db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE jobs SET status = ?, data = ? WHERE id = ?",
                (job.status, json.dumps(job.to_dict()), job.job_id)
            )
            
        # Mock callback
        callback = MagicMock(return_value=True)
        auto_resume_service.set_resume_callback(callback)
        
        # Trigger auto-resume
        auto_resume_service._try_auto_resume()
        
        # Verify callback was called
        callback.assert_called_once_with(job.job_id)
        
        # Verify retry count incremented
        updated_job = job_manager.get_job(job.job_id)
        assert updated_job.retry_count == 1

    def test_max_retries(self, auto_resume_service, job_manager, test_db):
        """Test that max retries is respected."""
        # Force clean DB
        with test_db.get_cursor() as cursor:
            cursor.execute("DELETE FROM jobs")
            
        job = job_manager.create_job("Max Retry Job", "Orpheus", "zac", "M4B")
        checkpoint = JobCheckpoint(total_lines=50, lines_completed=5)
        job.checkpoint = checkpoint.to_dict()
        job.status = JobStatus.STALLED.value
        job.retry_count = job_manager.MAX_AUTO_RETRIES + 1
        
        # Update DB directly to set retry_count
        with test_db.get_cursor() as cursor:
            # Verify to_dict
            data_json = json.dumps(job.to_dict())
            if "retry_count" not in data_json:
                pytest.fail("retry_count missing from to_dict JSON")
                
            cursor.execute(
                "UPDATE jobs SET status = ?, data = ? WHERE id = ?",
                (job.status, data_json, job.job_id)
            )
            
        callback = MagicMock()
        auto_resume_service.set_resume_callback(callback)
        
        auto_resume_service._try_auto_resume()
        
        callback.assert_not_called()
