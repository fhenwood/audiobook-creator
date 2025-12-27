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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        # Handle old jobs without new fields
        if 'checkpoint' not in data:
            data['checkpoint'] = None
        if 'last_activity' not in data:
            data['last_activity'] = data.get('updated_at')
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
    Singleton job manager that persists jobs for 24 hours.
    Jobs are stored in a JSON file and cleaned up automatically.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    JOBS_FILE = "generated_audiobooks/jobs.json"
    JOB_EXPIRY_HOURS = 24
    
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
        """Remove files associated with a job."""
        job = self._jobs.get(job_id)
        if job and job.output_file and os.path.exists(job.output_file):
            try:
                os.remove(job.output_file)
            except Exception as e:
                print(f"Error removing job file {job.output_file}: {e}")
    
    def create_job(
        self, 
        book_title: str, 
        tts_engine: str, 
        voice: str, 
        output_format: str
    ) -> Job:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())[:8]  # Short UUID for easier reference
        now = datetime.now()
        expires_at = now + timedelta(hours=self.JOB_EXPIRY_HOURS)
        
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
