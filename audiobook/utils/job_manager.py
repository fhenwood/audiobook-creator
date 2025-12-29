"""
Job Manager - Manages audiobook generation jobs that persist for 1 week.
Allows users to reconnect and retrieve their job results.
"""

import os
import json
import time
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"  # Job stopped unexpectedly but can be resumed


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
        self._jobs: Dict[str, Job] = {}
        self._load_jobs()
        self._cleanup_expired_jobs()
    
    def _load_jobs(self):
        """Load jobs from persistent storage."""
        try:
            if os.path.exists(self.JOBS_FILE):
                with open(self.JOBS_FILE, 'r') as f:
                    data = json.load(f)
                    self._jobs = {
                        job_id: Job.from_dict(job_data) 
                        for job_id, job_data in data.items()
                    }
        except Exception as e:
            print(f"Error loading jobs: {e}")
            self._jobs = {}
    
    def _save_jobs(self):
        """Save jobs to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.JOBS_FILE), exist_ok=True)
            with open(self.JOBS_FILE, 'w') as f:
                data = {job_id: job.to_dict() for job_id, job in self._jobs.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving jobs: {e}")
    
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
    
    def create_job_directory(self, job_id: str):
        """Create the working directory structure for a job."""
        job_dir = self.get_job_dir(job_id)
        line_segments_dir = self.get_job_line_segments_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        os.makedirs(line_segments_dir, exist_ok=True)
        return job_dir
    
    def _cleanup_expired_jobs(self):
        """Remove jobs that have expired (older than 1 week)."""
        now = datetime.now()
        expired_jobs = []
        
        for job_id, job in self._jobs.items():
            try:
                expires_at = datetime.fromisoformat(job.expires_at)
                if now > expires_at:
                    expired_jobs.append(job_id)
            except Exception:
                expired_jobs.append(job_id)
        
        for job_id in expired_jobs:
            self._remove_job_files(job_id)
            del self._jobs[job_id]
        
        if expired_jobs:
            self._save_jobs()
            print(f"Cleaned up {len(expired_jobs)} expired jobs")
    
    def _remove_job_files(self, job_id: str):
        """Remove all files associated with a job, including working directory."""
        job = self._jobs.get(job_id)
        
        # Remove output file (final audiobook)
        if job and job.output_file and os.path.exists(job.output_file):
            try:
                os.remove(job.output_file)
                print(f"Removed output file: {job.output_file}")
            except Exception as e:
                print(f"Error removing job file {job.output_file}: {e}")
        
        # Remove persistent book copy
        if job and job.checkpoint:
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
        output_format: str
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
        )
        
        self._jobs[job_id] = job
        self._save_jobs()
        return job
    
    def update_job_progress(self, job_id: str, progress: str):
        """Update job progress message."""
        if job_id in self._jobs:
            now = datetime.now().isoformat()
            self._jobs[job_id].progress = progress
            self._jobs[job_id].updated_at = now
            self._jobs[job_id].last_activity = now
            self._jobs[job_id].status = JobStatus.IN_PROGRESS.value
            self._save_jobs()
    
    def update_job_checkpoint(self, job_id: str, checkpoint: JobCheckpoint):
        """Update job checkpoint for resume capability."""
        if job_id in self._jobs:
            now = datetime.now().isoformat()
            self._jobs[job_id].checkpoint = checkpoint.to_dict()
            self._jobs[job_id].last_activity = now
            self._jobs[job_id].updated_at = now
            self._save_jobs()
    
    def mark_job_stalled(self, job_id: str):
        """Mark a job as stalled (can be resumed)."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == JobStatus.IN_PROGRESS.value and job.checkpoint:
                checkpoint = job.get_checkpoint()
                lines_done = checkpoint.lines_completed if checkpoint else 0
                total_lines = checkpoint.total_lines if checkpoint else 0
                pct = (lines_done / total_lines * 100) if total_lines > 0 else 0
                
                job.status = JobStatus.STALLED.value
                job.progress = f"⏸️ Stalled at {pct:.1f}% ({lines_done}/{total_lines} lines) - Click Resume to continue"
                job.updated_at = datetime.now().isoformat()
                self._save_jobs()
                return True
        return False
    
    def check_for_stalled_jobs(self, stall_timeout_seconds: int = 300):
        """Check for jobs that have stalled (no activity for timeout period)."""
        now = datetime.now()
        stalled_jobs = []
        
        for job_id, job in self._jobs.items():
            if job.status == JobStatus.IN_PROGRESS.value:
                try:
                    last_activity = datetime.fromisoformat(job.last_activity or job.updated_at)
                    elapsed = (now - last_activity).total_seconds()
                    if elapsed > stall_timeout_seconds:
                        if self.mark_job_stalled(job_id):
                            stalled_jobs.append(job_id)
                            print(f"⏸️ Job {job_id} marked as stalled (no activity for {elapsed:.0f}s)")
                except Exception as e:
                    print(f"Error checking job {job_id} for stall: {e}")
        
        return stalled_jobs
    
    MAX_AUTO_RETRIES = 3
    
    def can_auto_retry(self, job_id: str) -> bool:
        """Check if a stalled job can be auto-retried."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            return (
                job.status == JobStatus.STALLED.value and 
                job.checkpoint is not None and
                job.retry_count < self.MAX_AUTO_RETRIES
            )
        return False
    
    def increment_retry_count(self, job_id: str) -> int:
        """Increment retry count for a job and return the new count."""
        if job_id in self._jobs:
            self._jobs[job_id].retry_count += 1
            self._save_jobs()
            return self._jobs[job_id].retry_count
        return 0
    
    def reset_retry_count(self, job_id: str):
        """Reset retry count (called on successful completion)."""
        if job_id in self._jobs:
            self._jobs[job_id].retry_count = 0
            self._save_jobs()
    
    def get_jobs_needing_auto_retry(self) -> List[str]:
        """Get list of stalled job IDs that can be auto-retried."""
        return [
            job_id for job_id, job in self._jobs.items()
            if self.can_auto_retry(job_id)
        ]
    
    def prepare_job_for_resume(self, job_id: str) -> bool:
        """Prepare a stalled job for resumption."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == JobStatus.STALLED.value and job.checkpoint:
                job.status = JobStatus.PENDING.value
                job.progress = "🔄 Ready to resume..."
                job.updated_at = datetime.now().isoformat()
                self._save_jobs()
                return True
        return False
    
    def complete_job(self, job_id: str, output_file: str):
        """Mark a job as completed with the output file path."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.COMPLETED.value
            self._jobs[job_id].output_file = output_file
            self._jobs[job_id].progress = "✅ Completed"
            self._jobs[job_id].updated_at = datetime.now().isoformat()
            self._save_jobs()
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark a job as failed with an error message."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.FAILED.value
            self._jobs[job_id].error_message = error_message
            self._jobs[job_id].progress = f"❌ Failed: {error_message}"
            self._jobs[job_id].updated_at = datetime.now().isoformat()
            self._save_jobs()
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        self._cleanup_expired_jobs()
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[Job]:
        """Get all non-expired jobs, sorted by creation time (newest first)."""
        self._cleanup_expired_jobs()
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        if job_id in self._jobs:
            self._remove_job_files(job_id)
            del self._jobs[job_id]
            self._save_jobs()
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
            except:
                time_str = "Unknown"
            
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
        
        for job_id, job in job_manager._jobs.items():
            # Jobs that were in-progress when the server stopped should be marked as stalled
            if job.status == JobStatus.IN_PROGRESS.value:
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
    except:
        time_str = "?"
    
    # Format created time
    try:
        created = datetime.fromisoformat(job.created_at)
        created_str = created.strftime("%m/%d %H:%M")
    except:
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
