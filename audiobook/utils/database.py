
import sqlite3
import json
import os
import threading
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

class Database:
    """
    SQLite database wrapper for persistent job storage.
    Thread-safe connection handling.
    """
    
    # Use absolute path to avoid CWD issues
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_FILE = os.path.join(PROJECT_ROOT, "generated_audiobooks", "audiobook_jobs.db")
    
    def __init__(self):
        self._local = threading.local()
        self._ensure_db_directory()
        self._init_schema()

    def _ensure_db_directory(self):
        os.makedirs(os.path.dirname(self.DB_FILE), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.DB_FILE)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor with auto-commit."""
        conn = self._get_connection()
        try:
            yield conn.cursor()
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    def _init_schema(self):
        """Initialize the database schema."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    book_title TEXT,
                    progress TEXT,
                    percent_complete REAL DEFAULT 0.0,
                    data TEXT NOT NULL
                )
            """)
            # Index for faster lookups/sorting
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            
            # Migration: Ensure columns exist (for existing DBs)
            self._ensure_columns_exist(cursor, "jobs", ["expires_at", "progress", "percent_complete", "completed_lines"])

    def _ensure_columns_exist(self, cursor, table, columns):
        """Check if columns exist and add them if not."""
        cursor.execute(f"PRAGMA table_info({table})")
        existing_columns = {row['name'] for row in cursor.fetchall()}
        
        for col in columns:
            if col not in existing_columns:
                print(f"🔄 Migrating DB: Adding column '{col}' to table '{table}'")
                col_type = "TEXT"
                if col == "percent_complete":
                     col_type = "REAL DEFAULT 0.0"
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

    def close(self):
        """Close the thread-local connection (if any)."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

# Global instance
db = Database()
