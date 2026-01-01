"""
Text Loader - Handles loading and preparing text for audiobook generation.

Extracts text loading logic from generator.py for modularity.
"""

import os
from typing import List, Optional
from audiobook.utils.job_manager import job_manager
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


class TextLoader:
    """
    Loads and prepares text for audiobook generation.
    
    Handles:
    - Reading converted book text
    - Loading emotion-tagged text if available
    - Text normalization
    """
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._lines: List[str] = []

    def load_lines(self, file_path: str) -> List[str]:
        """
        Load lines from a specific file path.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List of text lines.
        """
        if not file_path or not os.path.exists(file_path):
             # Try fallback to load() if path not valid/empty provided?
             # Or raise error.
             # Pipeline passes config.book_path.
             if not file_path:
                 return self.load()
             raise FileNotFoundError(f"File not found: {file_path}")
             
        logger.info(f"Loading text from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self._lines = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(self._lines)} lines")
        return self._lines
    
    def load(self, add_emotion_tags: bool = False) -> List[str]:
        """
        Load text from the appropriate source.
        
        Args:
            add_emotion_tags: If True, prefer emotion-tagged text.
            
        Returns:
            List of text lines.
        """
        # Check for emotion-tagged text first
        emotion_tags_path = job_manager.get_job_emotion_tags_path(self.job_id)
        converted_book_path = job_manager.get_job_converted_book_path(self.job_id)
        
        text_file = None
        source = None
        
        if add_emotion_tags and os.path.exists(emotion_tags_path):
            text_file = emotion_tags_path
            source = "emotion_tags"
        elif os.path.exists(converted_book_path):
            text_file = converted_book_path
            source = "converted_book"
        
        # Fallback to legacy global paths (DEPRECATED - will be removed)
        if not text_file:
            if add_emotion_tags and os.path.exists("tag_added_lines_chunks.txt"):
                text_file = "tag_added_lines_chunks.txt"
                source = "legacy_emotion_tags"
                logger.warning(
                    f"⚠️ Using deprecated global path 'tag_added_lines_chunks.txt'. "
                    f"Copy to job directory: {emotion_tags_path}"
                )
            elif os.path.exists("converted_book.txt"):
                text_file = "converted_book.txt"
                source = "legacy_converted_book"
                logger.warning(
                    f"⚠️ Using deprecated global path 'converted_book.txt'. "
                    f"Copy to job directory: {converted_book_path}"
                )
        
        if not text_file:
            raise FileNotFoundError(f"No text file found for job {self.job_id}")
        
        logger.info(f"Loading text from {source}: {text_file}")
        
        with open(text_file, "r", encoding="utf-8") as f:
            self._lines = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self._lines)} lines")
        return self._lines
    
    @property
    def lines(self) -> List[str]:
        return self._lines
    
    @property
    def total_lines(self) -> int:
        return len(self._lines)
