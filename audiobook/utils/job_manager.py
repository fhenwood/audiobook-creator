"""
Job Manager - Manages audiobook generation jobs that persist for 1 week.
Allows users to reconnect and retrieve their job results.
"""

import os
import json
import time
import uuid
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from audiobook.utils.database import db
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"  # Job stopped unexpectedly but can be resumed

@dataclass
class JobProgress:
    """Structured progress update for a job."""
    message: str
    percent_complete: float = 0.0
    stage: str = "processing" 
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        return f"[{self.percent_complete:.1f}%] {self.message}"


@dataclass
class JobCheckpoint:
    """Checkpoint data for resumable jobs."""
    total_lines: int = 0
    lines_completed: int = 0
    current_chapter_index: int = 0
    chapter_files: List[str] = None
    chapter_line_map: Dict[str, List[int]] = None
    book_file_path: str = ""
    add_emotion_tags: bool = False
    reference_audio_path: str = ""
    temp_audio_dir: str = "temp_audio"
    
    def __post_init__(self):
        if self.chapter_files is None:
            self.chapter_files = []
        if self.chapter_line_map is None:
            self.chapter_line_map = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobCheckpoint":
        if data is None:
            return None
        return cls(**data)


@dataclass
class Job:
    """Represents an audiobook generation job."""
    job_id: str
    book_title: str
    tts_engine: str
    voice: str
    output_format: str
    status: str
    progress: str
    created_at: str
    updated_at: str
    expires_at: str
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    # Checkpoint support for resumable jobs
    checkpoint: Optional[Dict[str, Any]] = None
    last_activity: Optional[str] = None  # ISO timestamp of last activity
    retry_count: int = 0  # Number of auto-resume attempts for stalled jobs
    percent_complete: float = 0.0  # Percentage complete (0-100)
    # New VibeVoice / Post-processing fields
    postprocess: bool = False
    vibevoice_voice: Optional[str] = None
    vibevoice_temperature: float = 0.7
    vibevoice_top_p: float = 0.95
    use_vibevoice_dialogue: bool = False
    vibevoice_dialogue_voice: Optional[str] = None
    verification_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        # Handle old jobs without new fields
        if 'checkpoint' not in data:
            data['checkpoint'] = None
        if 'last_activity' not in data:
            data['last_activity'] = data.get('updated_at')
        if 'retry_count' not in data:
            data['retry_count'] = 0
        if 'percent_complete' not in data:
            data['percent_complete'] = 0.0
        if 'verification_enabled' not in data:
            data['verification_enabled'] = False
        return cls(**data)
    
    def get_checkpoint(self) -> Optional[JobCheckpoint]:
        """Get checkpoint as a JobCheckpoint object."""
        if self.checkpoint:
            return JobCheckpoint.from_dict(self.checkpoint)
        return None
    
    def is_resumable(self) -> bool:
        """Check if this job can be resumed."""
        return (
            self.status == JobStatus.STALLED.value and 
            self.checkpoint is not None and
            self.checkpoint.get('lines_completed', 0) > 0
        )


class JobManager:
    """
    Singleton job manager that persists jobs for 1 week.
    Jobs are stored in a JSON file and cleaned up automatically.
    Each job has its own working directory under temp_audio/{job_id}/
    """
    
    _instance = None
    _lock = threading.Lock()
    
    JOBS_FILE = "generated_audiobooks/jobs.json"
    JOB_EXPIRY_HOURS = 168  # 1 week
    TEMP_AUDIO_BASE = "temp_audio"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # self._jobs removed in favor of DB
        self._migrate_json_to_db()
        self._cleanup_expired_jobs()
    
    def _migrate_json_to_db(self):
        """Migrate legacy jobs.json to SQLite."""
        if os.path.exists(self.JOBS_FILE):
            logger.info("🔄 Migrating jobs.json to SQLite database...")
            try:
                with open(self.JOBS_FILE, 'r') as f:
                    data = json.load(f)
                    
                count = 0
                with db.get_cursor() as cursor:
                    for job_id, job_data in data.items():
                        # Check if exists
                        cursor.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,))
                        if cursor.fetchone():
                            continue
                            
                        job = Job.from_dict(job_data)
                        cursor.execute(
                            "INSERT INTO jobs (id, created_at, updated_at, expires_at, status, book_title, percent_complete, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                job.job_id,
                                job.created_at,
                                job.updated_at,
                                job.expires_at,
                                job.status,
                                job.book_title,
                                job.percent_complete,
                                json.dumps(job.to_dict())
                            )
                        )
                        count += 1
                
                logger.info(f"✅ Migrated {count} jobs to database.")
                os.rename(self.JOBS_FILE, self.JOBS_FILE + ".bak")
                logger.info(f"Renamed {self.JOBS_FILE} to {self.JOBS_FILE}.bak")
                
            except Exception as e:
                logger.error(f"❌ Error migrating jobs: {e}")
    
    def _save_jobs(self):
        """No-op for backward compatibility if called internally."""
        pass
    
    def get_job_dir(self, job_id: str) -> str:
        """Get the working directory for a job."""
        return os.path.join(self.TEMP_AUDIO_BASE, job_id)
    
    def get_job_line_segments_dir(self, job_id: str) -> str:
        """Get the line segments directory for a job."""
        return os.path.join(self.get_job_dir(job_id), "line_segments")
    
    def get_job_converted_book_path(self, job_id: str) -> str:
        """Get the path to the converted book text file for a job."""
        return os.path.join(self.get_job_dir(job_id), "converted_book.txt")
    
    def get_job_emotion_tags_path(self, job_id: str) -> str:
        """Get the path to the emotion tags file for a job."""
        return os.path.join(self.get_job_dir(job_id), "tag_added_lines_chunks.txt")
    
    def get_job_cover_path(self, job_id: str) -> str:
        """Get the path to the cover image for a job."""
        return os.path.join(self.get_job_dir(job_id), "cover.jpg")
    
    def get_job_chapters_file_path(self, job_id: str) -> str:
        """Get the path to the chapters metadata file for a job."""
        return os.path.join(self.get_job_dir(job_id), "chapters.txt")
    
    def get_job_chapter_list_path(self, job_id: str) -> str:
        """Get the path to the chapter list file for a job."""
        return os.path.join(self.get_job_dir(job_id), "chapter_list.txt")
    
    def create_job_directory(self, job_id: str):
        """Create the working directory structure for a job."""
        job_dir = self.get_job_dir(job_id)
        line_segments_dir = self.get_job_line_segments_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        os.makedirs(line_segments_dir, exist_ok=True)
        return job_dir
    
    def _cleanup_expired_jobs(self):
        """Remove jobs that have expired."""
        now = datetime.now().isoformat()
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT id, data FROM jobs WHERE expires_at < ?", (now,))
                expired_rows = cursor.fetchall()
                
                for row in expired_rows:
                    self._remove_job_files(row['id'], json.loads(row['data']))
                
                if expired_rows:
                    cursor.execute("DELETE FROM jobs WHERE expires_at < ?", (now,))
                    logger.info(f"Cleaned up {len(expired_rows)} expired jobs")
        except Exception as e:
            logger.error(f"Error cleaning up expired jobs: {e}")
    
    def _remove_job_files(self, job_id: str, job_data: Dict = None):
        """Remove all files associated with a job."""
        if job_data is None:
            # Fetch if not provided
            try:
                with db.get_cursor() as cursor:
                    cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
                    row = cursor.fetchone()
                    if row:
                        job_data = json.loads(row['data'])
            except Exception:
                pass
        
        if not job_data:
            return

        job = Job.from_dict(job_data)
        
        # Remove output file (final audiobook)
        if job.output_file and os.path.exists(job.output_file):
            try:
                os.remove(job.output_file)
                print(f"Removed output file: {job.output_file}")
            except Exception as e:
                print(f"Error removing job file {job.output_file}: {e}")
        
        # Remove persistent book copy
        if job.checkpoint:
            book_path = job.checkpoint.get('book_file_path', '')
            if book_path and '_temp_' in book_path and os.path.exists(book_path):
                try:
                    os.remove(book_path)
                    print(f"Removed persistent book copy: {book_path}")
                except Exception as e:
                    print(f"Error removing book file {book_path}: {e}")
        
        # Remove job working directory (temp_audio/{job_id}/)
        job_dir = self.get_job_dir(job_id)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                print(f"Removed job directory: {job_dir}")
            except Exception as e:
                print(f"Error removing job directory {job_dir}: {e}")
    
    def create_job(
        self, 
        book_title: str, 
        tts_engine: str, 
        voice: str, 
        output_format: str,
        postprocess: bool = False,
        vibevoice_voice: str = None,
        vibevoice_temperature: float = 0.7,
        vibevoice_top_p: float = 0.95,
        use_vibevoice_dialogue: bool = False,
        vibevoice_dialogue_voice: str = None,
        verification_enabled: bool = False
    ) -> Job:
        """Create a new job, its working directory, and return the job."""
        job_id = str(uuid.uuid4())[:8]  # Short UUID for easier reference
        now = datetime.now()
        expires_at = now + timedelta(hours=self.JOB_EXPIRY_HOURS)
        
        # Create job working directory
        self.create_job_directory(job_id)
        
        job = Job(
            job_id=job_id,
            book_title=book_title,
            tts_engine=tts_engine,
            voice=voice,
            output_format=output_format,
            status=JobStatus.PENDING.value,
            progress="Waiting to start...",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            postprocess=postprocess,
            vibevoice_voice=vibevoice_voice,
            vibevoice_temperature=vibevoice_temperature,
            vibevoice_top_p=vibevoice_top_p,
            use_vibevoice_dialogue=use_vibevoice_dialogue,
            vibevoice_dialogue_voice=vibevoice_dialogue_voice,
            verification_enabled=verification_enabled
        )
        
        with db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO jobs (id, created_at, updated_at, expires_at, status, book_title, percent_complete, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.created_at,
                    job.updated_at,
                    job.expires_at,
                    job.status,
                    job.book_title,
                    job.percent_complete,
                    json.dumps(job.to_dict())
                )
            )
        return job
    
    def update_job_progress(self, job_id: str, progress: str, percent_complete: float = None):
        """Update job progress message and optionally percentage."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                now = datetime.now().isoformat()
                job.progress = progress
                if percent_complete is not None:
                    job.percent_complete = percent_complete
                job.updated_at = now
                job.last_activity = now
                job.status = JobStatus.IN_PROGRESS.value
                
                cursor.execute(
                    """
                    UPDATE jobs 
                    SET progress = ?, percent_complete = ?, updated_at = ?, status = ?, data = ?
                    WHERE id = ?
                    """,
                    (
                        job.progress, 
                        job.percent_complete, 
                        job.updated_at, 
                        job.status, 
                        json.dumps(job.to_dict()), 
                        job_id
                    )
                )
    
    def update_job_checkpoint(self, job_id: str, checkpoint: JobCheckpoint):
        """Update job checkpoint for resume capability."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                now = datetime.now().isoformat()
                job.checkpoint = checkpoint.to_dict()
                job.last_activity = now
                job.updated_at = now
                
                cursor.execute(
                    "UPDATE jobs SET updated_at = ?, data = ? WHERE id = ?",
                    (job.updated_at, json.dumps(job.to_dict()), job_id)
                )
    
    def mark_line_complete(self, job_id: str, line_index: int):
        """Mark a specific line as completed in the DB.
        
        This is the source of truth for resume logic - not filesystem checks.
        """
        with db.get_cursor() as cursor:
            cursor.execute("SELECT completed_lines FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                completed = json.loads(row['completed_lines'] or '[]')
                if line_index not in completed:
                    completed.append(line_index)
                    cursor.execute(
                        "UPDATE jobs SET completed_lines = ? WHERE id = ?",
                        (json.dumps(completed), job_id)
                    )
    
    def get_completed_lines(self, job_id: str) -> set:
        """Get set of completed line indices from DB.
        
        Returns:
            Set of line indices that have been successfully processed.
        """
        with db.get_cursor() as cursor:
            cursor.execute("SELECT completed_lines FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row and row['completed_lines']:
                return set(json.loads(row['completed_lines']))
        return set()
    
    def mark_job_stalled(self, job_id: str):
        """Mark a job as stalled (can be resumed)."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data, status FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return False
                
            job_data = json.loads(row['data'])
            job = Job.from_dict(job_data)
            
            if job.status == JobStatus.IN_PROGRESS.value and job.checkpoint:
                checkpoint = job.get_checkpoint()
                lines_done = checkpoint.lines_completed if checkpoint else 0
                total_lines = checkpoint.total_lines if checkpoint else 0
                pct = (lines_done / total_lines * 100) if total_lines > 0 else 0
                
                job.status = JobStatus.STALLED.value
                job.progress = f"⏸️ Stalled at {pct:.1f}% ({lines_done}/{total_lines} lines) - Click Resume to continue"
                job.updated_at = datetime.now().isoformat()
                
                cursor.execute(
                    "UPDATE jobs SET status = ?, progress = ?, updated_at = ?, data = ? WHERE id = ?",
                    (job.status, job.progress, job.updated_at, json.dumps(job.to_dict()), job_id)
                )
                return True
        return False
    
    def get_pending_jobs(self, limit: int = 10) -> List[Job]:
        """
        Get pending jobs ordered by creation time.
        
        Args:
            limit: Maximum number of jobs to return.
            
        Returns:
            List of pending Job objects.
        """
        jobs = []
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT data FROM jobs 
                WHERE status = ? 
                ORDER BY created_at ASC 
                LIMIT ?
                """,
                (JobStatus.PENDING.value, limit)
            )
            rows = cursor.fetchall()
            
            for row in rows:
                job_data = json.loads(row['data'])
                jobs.append(Job.from_dict(job_data))
        
        return jobs
    
    def check_for_stalled_jobs(self, stall_timeout_seconds: int = 300):
        """Check for jobs that have stalled (no activity for timeout period)."""
        now = datetime.now()
        stalled_jobs = []
        
        with db.get_cursor() as cursor:
            cursor.execute("SELECT id, data FROM jobs WHERE status = ?", (JobStatus.IN_PROGRESS.value,))
            rows = cursor.fetchall()
            
            for row in rows:
                job_id = row['id']
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                try:
                    last_activity = datetime.fromisoformat(job.last_activity or job.updated_at)
                    elapsed = (now - last_activity).total_seconds()
                    if elapsed > stall_timeout_seconds:
                        if self.mark_job_stalled(job_id):
                            stalled_jobs.append(job_id)
                            logger.info(f"⏸️ Job {job_id} marked as stalled (no activity for {elapsed:.0f}s)")
                except Exception as e:
                    logger.warning(f"Error checking job {job_id} for stall: {e}")
        
        return stalled_jobs
    
    MAX_AUTO_RETRIES = 3
    
    def can_auto_retry(self, job_id: str) -> bool:
        """Check if a stalled job can be auto-retried."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                return (
                    job.status == JobStatus.STALLED.value and 
                    job.checkpoint is not None and
                    job.retry_count < self.MAX_AUTO_RETRIES
                )
        return False
    
    def increment_retry_count(self, job_id: str) -> int:
        """Increment retry count for a job and return the new count."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                job.retry_count += 1
                cursor.execute(
                    "UPDATE jobs SET data = ? WHERE id = ?",
                    (json.dumps(job.to_dict()), job_id)
                )
                return job.retry_count
        return 0
    
    def reset_retry_count(self, job_id: str):
        """Reset retry count (called on successful completion)."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                job.retry_count = 0
                cursor.execute(
                    "UPDATE jobs SET data = ? WHERE id = ?",
                    (json.dumps(job.to_dict()), job_id)
                )
    
    def get_jobs_needing_auto_retry(self) -> List[str]:
        """Get list of stalled job IDs that can be auto-retried."""
        # This iterates all jobs which is inefficient, but okay for MVP. 
        # Better: SELECT id FROM jobs WHERE status='stalled' AND retry_count < MAX
        # but retry_count is in JSON blob only (unless I add column).
        # For now, fetch all stalled jobs and check.
        jobs_to_retry = []
        with db.get_cursor() as cursor:
            cursor.execute("SELECT id, data FROM jobs WHERE status = ?", (JobStatus.STALLED.value,))
            rows = cursor.fetchall()
            for row in rows:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                if job.checkpoint and job.retry_count < self.MAX_AUTO_RETRIES:
                    jobs_to_retry.append(job.job_id)
        return jobs_to_retry
    
    def prepare_job_for_resume(self, job_id: str) -> bool:
        """Prepare a stalled job for resumption."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                if job.status == JobStatus.STALLED.value and job.checkpoint:
                    job.status = JobStatus.PENDING.value
                    job.progress = "🔄 Ready to resume..."
                    job.updated_at = datetime.now().isoformat()
                    
                    cursor.execute(
                        "UPDATE jobs SET status = ?, progress = ?, updated_at = ?, data = ? WHERE id = ?",
                        (job.status, job.progress, job.updated_at, json.dumps(job.to_dict()), job_id)
                    )
                    return True
        return False
    
    def complete_job(self, job_id: str, output_file: str):
        """Mark a job as completed with the output file path."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                job.status = JobStatus.COMPLETED.value
                job.output_file = output_file
                job.progress = "✅ Completed"
                job.percent_complete = 100.0
                job.updated_at = datetime.now().isoformat()
                
                cursor.execute(
                    """
                    UPDATE jobs 
                    SET status = ?, percent_complete = ?, updated_at = ?, data = ? 
                    WHERE id = ?
                    """,
                    (job.status, job.percent_complete, job.updated_at, json.dumps(job.to_dict()), job_id)
                )
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed with an error message."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                job_data = json.loads(row['data'])
                job = Job.from_dict(job_data)
                
                job.status = JobStatus.FAILED.value
                job.error_message = error_message
                job.progress = f"❌ Failed: {error_message}"
                job.updated_at = datetime.now().isoformat()
                
                cursor.execute(
                    "UPDATE jobs SET status = ?, progress = ?, updated_at = ?, data = ? WHERE id = ?",
                    (job.status, job.progress, job.updated_at, json.dumps(job.to_dict()), job_id)
                )
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        self._cleanup_expired_jobs()
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                try:
                    return Job.from_dict(json.loads(row['data']))
                except Exception as e:
                    logger.error(f"Error parsing job data for {job_id}: {e}")
        return None
    
    def get_all_jobs(self) -> List[Job]:
        """Get all non-expired jobs, sorted by creation time (newest first)."""
        self._cleanup_expired_jobs()
        jobs = []
        with db.get_cursor() as cursor:
            cursor.execute("SELECT data FROM jobs ORDER BY created_at DESC")
            rows = cursor.fetchall()
            for row in rows:
                try:
                    jobs.append(Job.from_dict(json.loads(row['data'])))
                except Exception:
                    pass
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        # Fetch job data first to remove files
        with db.get_cursor() as cursor:
             cursor.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
             row = cursor.fetchone()
             if row:
                 try:
                     job_data = json.loads(row['data'])
                     self._remove_job_files(job_id, job_data)
                 except Exception:
                     # Even if parsing fails, delete from DB
                     self._remove_job_files(job_id)
                 
                 cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                 return True
        return False
    
    def get_job_summary(self) -> str:
        """Get a summary of all jobs for display."""
        jobs = self.get_all_jobs()
        if not jobs:
            return "No jobs found."
        
        lines = []
        for job in jobs:
            status_emoji = {
                JobStatus.PENDING.value: "⏳",
                JobStatus.IN_PROGRESS.value: "🔄",
                JobStatus.COMPLETED.value: "✅",
                JobStatus.FAILED.value: "❌",
                JobStatus.STALLED.value: "⏸️"
            }.get(job.status, "❓")
            
            # Calculate time remaining
            try:
                expires_at = datetime.fromisoformat(job.expires_at)
                remaining = expires_at - datetime.now()
                hours_remaining = max(0, remaining.total_seconds() / 3600)
                time_str = f"{hours_remaining:.1f}h remaining"
            except ValueError:
                time_str = "?"
            
            lines.append(
                f"{status_emoji} **{job.book_title}** (ID: `{job.job_id}`)\n"
                f"   Engine: {job.tts_engine} | Voice: {job.voice} | Format: {job.output_format}\n"
                f"   Status: {job.progress} | Expires: {time_str}"
            )
        
        return "\n\n".join(lines)


# Global instance
job_manager = JobManager()


class AutoResumeService:
    """
    Background service that monitors for stalled jobs and auto-resumes them.
    This runs in a separate thread and checks periodically for jobs that need resumption.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration
    CHECK_INTERVAL_SECONDS = 60  # How often to check for stalled jobs
    STALL_TIMEOUT_SECONDS = 300  # How long without activity before a job is considered stalled
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._resume_callback = None
        self._stop_event = threading.Event()
    
    def set_resume_callback(self, callback):
        """
        Set the callback function to resume a job.
        The callback should accept a job_id and return True if resume was started.
        """
        self._resume_callback = callback
    
    def start(self):
        """Start the auto-resume monitoring thread."""
        if self._running:
            print("⚠️ Auto-resume service already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="auto-resume")
        self._thread.start()
        print(f"🔄 Auto-resume service started (checking every {self.CHECK_INTERVAL_SECONDS}s)")
    
    def stop(self):
        """Stop the auto-resume monitoring thread."""
        if not self._running:
            return
        
        print("🛑 Stopping auto-resume service...")
        self._stop_event.set()
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        print("✅ Auto-resume service stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a background thread."""
        print("🔍 Auto-resume monitor started")
        
        # Initial check on startup - detect and mark stalled jobs
        self._check_and_mark_stalled_jobs()
        
        while not self._stop_event.is_set():
            try:
                # Wait for the check interval or until stopped
                if self._stop_event.wait(timeout=self.CHECK_INTERVAL_SECONDS):
                    break  # Stop event was set
                
                # Check for stalled jobs
                self._check_and_mark_stalled_jobs()
                
                # Try to auto-resume eligible jobs
                self._try_auto_resume()
                
            except Exception as e:
                print(f"❌ Error in auto-resume monitor: {e}")
                import traceback
                traceback.print_exc()
    
    def _check_and_mark_stalled_jobs(self):
        """Check for jobs that have stalled and mark them."""
        try:
            stalled = job_manager.check_for_stalled_jobs(self.STALL_TIMEOUT_SECONDS)
            if stalled:
                print(f"⏸️ Marked {len(stalled)} job(s) as stalled: {stalled}")
        except Exception as e:
            print(f"❌ Error checking for stalled jobs: {e}")
    
    def _try_auto_resume(self):
        """Try to auto-resume jobs that are eligible."""
        if not self._resume_callback:
            return
        
        try:
            jobs_to_retry = job_manager.get_jobs_needing_auto_retry()
            
            for job_id in jobs_to_retry:
                job = job_manager.get_job(job_id)
                if not job:
                    continue
                
                retry_count = job_manager.increment_retry_count(job_id)
                print(f"🔄 Auto-resuming job {job_id} (attempt {retry_count}/{job_manager.MAX_AUTO_RETRIES})")
                
                try:
                    # Call the resume callback
                    if self._resume_callback(job_id):
                        print(f"✅ Auto-resume started for job {job_id}")
                    else:
                        print(f"⚠️ Auto-resume callback returned False for job {job_id}")
                except Exception as e:
                    print(f"❌ Error auto-resuming job {job_id}: {e}")
                    job_manager.fail_job(job_id, f"Auto-resume failed: {str(e)}")
                
                # Only try one job at a time to avoid resource conflicts
                break
                
        except Exception as e:
            print(f"❌ Error in auto-resume: {e}")
    
    def check_startup_stalled_jobs(self) -> List[str]:
        """
        Check for stalled jobs on startup and return their IDs.
        This is called once at startup to identify jobs that stalled during the last run.
        """
        stalled_job_ids = []
        
        with db.get_cursor() as cursor:
            cursor.execute("SELECT id FROM jobs WHERE status = ?", (JobStatus.IN_PROGRESS.value,))
            rows = cursor.fetchall()
            
            for row in rows:
                job_id = row['id']
                if job_manager.mark_job_stalled(job_id):
                    stalled_job_ids.append(job_id)
                    print(f"⏸️ Startup: Marked job {job_id} as stalled (was in-progress)")
        
        return stalled_job_ids


# Global auto-resume service instance
auto_resume_service = AutoResumeService()


def format_job_for_table(job: Job) -> List[Any]:
    """Format a job for display in a Gradio dataframe."""
    status_emoji = {
        JobStatus.PENDING.value: "⏳ Pending",
        JobStatus.IN_PROGRESS.value: "🔄 In Progress",
        JobStatus.COMPLETED.value: "✅ Completed",
        JobStatus.FAILED.value: "❌ Failed",
        JobStatus.STALLED.value: "⏸️ Stalled (Resumable)"
    }.get(job.status, job.status)
    
    # Calculate time remaining
    try:
        expires_at = datetime.fromisoformat(job.expires_at)
        remaining = expires_at - datetime.now()
        hours_remaining = max(0, remaining.total_seconds() / 3600)
        time_str = f"{hours_remaining:.1f}h"
    except ValueError:
        time_str = "?"
    
    # Format created time
    try:
        created = datetime.fromisoformat(job.created_at)
        created_str = created.strftime("%m/%d %H:%M")
    except ValueError:
        created_str = "?"
    
    return [
        job.job_id,
        job.book_title,
        job.tts_engine,
        job.voice,
        job.output_format,
        status_emoji,
        job.progress[:50] + "..." if len(job.progress) > 50 else job.progress,
        created_str,
        time_str
    ]


def get_jobs_dataframe() -> List[List[Any]]:
    """Get all jobs formatted for a Gradio dataframe."""
    jobs = job_manager.get_all_jobs()
    return [format_job_for_table(job) for job in jobs]
