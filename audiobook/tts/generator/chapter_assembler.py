"""
Chapter Assembler - Assembles line audio files into chapters.

Handles:
- Organizing lines into chapters based on chapter headings
- FFmpeg-based audio concatenation (memory-efficient)
- Post-processing (silence, format conversion)
"""

import os
import shutil
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

from audiobook.tts.generator.utils import check_if_chapter_heading, sanitize_filename
from audiobook.tts.audio_utils import (
    assemble_chapter_with_ffmpeg,
    add_silence_to_chapter_with_ffmpeg,
    convert_audio_file_formats,
    merge_chapters_to_m4b,
    merge_chapters_to_standard_audio_file,
)
from audiobook.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChapterInfo:
    """Information about a chapter."""
    name: str
    filename: str
    line_indices: List[int] = field(default_factory=list)


class ChapterAssembler:
    """
    Assembles line audio files into chapters and final audiobook.
    
    Usage:
        assembler = ChapterAssembler(temp_audio_dir, line_segments_dir)
        
        # Organize lines into chapters
        chapters = assembler.organize_chapters(lines, completed_indices)
        
        # Assemble chapter files
        await assembler.assemble_chapters(chapters)
        
        # Create final audiobook
        output_path = await assembler.finalize(book_path, output_format)
    """
    
    def __init__(
        self,
        temp_audio_dir: str,
        line_segments_dir: str,
        silence_duration_ms: int = 1000,
    ):
        self.temp_audio_dir = temp_audio_dir
        self.line_segments_dir = line_segments_dir
        self.silence_duration_ms = silence_duration_ms
        self._chapter_files: List[str] = []
    
    def organize_chapters(
        self,
        lines: List[str],
        completed_indices: Set[int]
    ) -> List[ChapterInfo]:
        """
        Organize lines into chapters based on chapter headings.
        
        Args:
            lines: All text lines from the book.
            completed_indices: Set of line indices that have audio files.
            
        Returns:
            List of ChapterInfo objects.
        """
        chapters: List[ChapterInfo] = []
        current_chapter = ChapterInfo(
            name="Introduction",
            filename="Introduction.wav"
        )
        
        for i, line in enumerate(lines):
            if i not in completed_indices:
                if i < 5: logger.debug(f"Line {i} not in completed_indices. indices sample: {list(completed_indices)[:5]}")
                continue  # Skip lines that don't have audio
            
            # Verify audio file exists
            line_audio_path = os.path.join(
                self.line_segments_dir, 
                f"line_{i:06d}.wav"
            )
            if not os.path.exists(line_audio_path):
                logger.warning(f"Audio file missing for line {i} at {line_audio_path}")
                continue
            
            # Check if this line starts a new chapter
            if check_if_chapter_heading(line):
                # Save current chapter if it has lines
                if current_chapter.line_indices:
                    chapters.append(current_chapter)
                
                # Start new chapter
                chapter_name = sanitize_filename(line)
                current_chapter = ChapterInfo(
                    name=line,
                    filename=f"{chapter_name}.wav"
                )
            
            current_chapter.line_indices.append(i)
        
        # Don't forget the last chapter
        if current_chapter.line_indices:
            chapters.append(current_chapter)
        else:
             logger.warning("Last chapter has no lines!")
        
        logger.info(f"Organized {len(chapters)} chapters from {len(completed_indices)} lines")
        return chapters
    
    async def assemble_chapters(
        self,
        chapters: List[ChapterInfo],
        use_postprocessing: bool = False
    ) -> List[str]:
        """
        Assemble line audio files into chapter files.
        
        Args:
            chapters: List of ChapterInfo from organize_chapters.
            use_postprocessing: Whether to apply audio enhancement.
            
        Returns:
            List of chapter filenames.
        """
        self._chapter_files = []
        
        for i, chapter in enumerate(chapters):
            logger.info(f"Assembling chapter {i+1}/{len(chapters)}: {chapter.name[:50]}...")
            
            # Use FFmpeg-based assembly (memory efficient)
            assemble_chapter_with_ffmpeg(
                chapter.filename,
                chapter.line_indices,
                self.line_segments_dir,
                self.temp_audio_dir
            )
            
            chapter_path = os.path.join(self.temp_audio_dir, chapter.filename)
            
            # Add silence between chapters
            if os.path.exists(chapter_path) and self.silence_duration_ms > 0:
                add_silence_to_chapter_with_ffmpeg(chapter_path, self.silence_duration_ms)
            
            # Optional post-processing
            if use_postprocessing and os.path.exists(chapter_path):
                await self._apply_postprocessing(chapter_path)
            
            self._chapter_files.append(chapter.filename)
        
        return self._chapter_files
    
    async def convert_chapters_to_m4a(self) -> List[str]:
        """Convert all WAV chapter files to M4A format."""
        m4a_files = []
        
        for chapter_wav in self._chapter_files:
            chapter_name = os.path.splitext(chapter_wav)[0]
            m4a_filename = f"{chapter_name}.m4a"
            
            convert_audio_file_formats(
                "wav", "m4a",
                self.temp_audio_dir,
                chapter_name
            )
            m4a_files.append(m4a_filename)
        
        return m4a_files
    
    async def finalize(
        self,
        book_path: Optional[str],
        output_format: str = "m4b",
        output_dir: str = "generated_audiobooks"
    ) -> str:
        """
        Create the final audiobook file.
        
        Args:
            book_path: Path to original book (for M4B metadata/cover).
            output_format: Desired output format.
            output_dir: Where to save the final audiobook.
            
        Returns:
            Path to the created audiobook file.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert WAV chapters to M4A
        m4a_files = await self.convert_chapters_to_m4a()
        
        if output_format.lower() == "m4b" and book_path:
            merge_chapters_to_m4b(book_path, m4a_files, self.temp_audio_dir)
            output_path = os.path.join(output_dir, "audiobook.m4b")
        else:
            merge_chapters_to_standard_audio_file(m4a_files, self.temp_audio_dir)
            
            # Convert to desired format if not m4a
            if output_format.lower() != "m4a":
                convert_audio_file_formats(
                    "m4a", output_format.lower(),
                    output_dir, "audiobook"
                )
            output_path = os.path.join(output_dir, f"audiobook.{output_format.lower()}")
        
        logger.info(f"Created audiobook: {output_path}")
        return output_path
    
    async def _apply_postprocessing(self, audio_path: str):
        """Apply audio enhancement to a chapter."""
        try:
            from audiobook.utils.audio_enhancer import audio_pipeline
            
            processed_path, status_msg = audio_pipeline.process(
                audio_path,
                output_path=audio_path,
                enable_preprocessing=True,
                stages=['enhancement', 'sr']
            )
            
            if processed_path:
                logger.info(f"Enhanced: {status_msg}")
            else:
                logger.warning(f"Enhancement failed for {audio_path}")
                
        except ImportError:
            logger.warning("Audio enhancer not available, skipping post-processing")
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
    
    def cleanup_line_segments(self):
        """Remove temporary line audio files."""
        if os.path.exists(self.line_segments_dir):
            shutil.rmtree(self.line_segments_dir)
            logger.info("Cleaned up line segment files")
    
    @property
    def chapter_files(self) -> List[str]:
        return self._chapter_files
