
import os
import shutil
import unittest
import time
from audiobook.utils.database import db
from audiobook.utils.job_manager import JobManager, JobStatus, Job, JobProgress, JobCheckpoint

class TestJobManagerDB(unittest.TestCase):
    
    def setUp(self):
        # Use a separate test DB if possible, or clean up the main one
        # For now, we'll use the main one but clean up jobs created
        self.manager = JobManager()
        self.created_jobs = []

    def tearDown(self):
        for job_id in self.created_jobs:
            self.manager.delete_job(job_id)

    def test_create_and_get_job(self):
        job = self.manager.create_job("Test Book", "Orpheus", "en-US", "M4B")
        self.created_jobs.append(job.job_id)
        
        self.assertIsNotNone(job.job_id)
        self.assertEqual(job.book_title, "Test Book")
        self.assertEqual(job.status, JobStatus.PENDING.value)
        
        # Determine if fetching from DB works
        fetched_job = self.manager.get_job(job.job_id)
        self.assertIsNotNone(fetched_job)
        self.assertEqual(fetched_job.job_id, job.job_id)
        self.assertEqual(fetched_job.book_title, "Test Book")

    def test_update_progress(self):
        job = self.manager.create_job("Progress Test", "Orpheus", "en-US", "M4B")
        self.created_jobs.append(job.job_id)
        
        self.manager.update_job_progress(job.job_id, "Processing...", 50.0)
        
        fetched_job = self.manager.get_job(job.job_id)
        self.assertEqual(fetched_job.progress, "Processing...")
        self.assertEqual(fetched_job.percent_complete, 50.0)
        self.assertEqual(fetched_job.status, JobStatus.IN_PROGRESS.value)

    def test_list_jobs(self):
        job1 = self.manager.create_job("List Test 1", "Orpheus", "en-US", "M4B")
        self.created_jobs.append(job1.job_id)
        time.sleep(1) # Ensure timestamp diff
        job2 = self.manager.create_job("List Test 2", "Orpheus", "en-US", "M4B")
        self.created_jobs.append(job2.job_id)
        
        jobs = self.manager.get_all_jobs()
        ids = [j.job_id for j in jobs]
        self.assertIn(job1.job_id, ids)
        self.assertIn(job2.job_id, ids)
        
        # Verify order (newest first)
        # This job2 should be before job1 if sorted by created_at DESC
        # Need to find their indices
        idx1 = ids.index(job1.job_id)
        idx2 = ids.index(job2.job_id)
        self.assertTrue(idx2 < idx1, f"Job 2 (newer) should be before Job 1 (older). Found indices {idx2}, {idx1}")

if __name__ == '__main__':
    unittest.main()
