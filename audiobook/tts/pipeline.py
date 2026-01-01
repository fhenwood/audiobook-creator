"""
Audiobook Pipeline - Orchestrates the audiobook generation process.

This module provides a clean, modular interface for generating audiobooks
by coordinating text loading, TTS generation, and audio assembly.

This is the PRIMARY ENTRY POINT for all audiobook generation - called by
both the Gradio UI and REST API.
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Set, AsyncGenerator, Dict, Any
from enum import Enum

from audiobook.utils.job_manager import job_manager, JobProgress, JobCheckpoint
from audiobook.utils.logging_config import get_logger
from audiobook.tts.generator.text_loader import TextLoader
from audiobook.tts.generator.line_generator import LineGenerator
from audiobook.tts.generator.line_generator import LineGenerator
from audiobook.tts.generator.chapter_assembler import ChapterAssembler
from audiobook.core.verification import TranscriptionVerifier
from audiobook.utils.gpu_resource_manager import gpu_manager
from audiobook.tts.service import tts_service

logger = get_logger(__name__)

# Constants
BATCH_SIZE = 50  # Batch size for all engines (keep large to minimize load/unload overhead)
PROGRESS_UPDATE_INTERVAL = 5  # Update UI progress every N lines within a batch


class PipelineStage(str, Enum):
    """Stages of the audiobook generation pipeline."""
    INITIALIZING = "initializing"
    LOADING_TEXT = "loading_text"
    GENERATING_AUDIO = "generating"
    ASSEMBLING_CHAPTERS = "assembling"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the audiobook pipeline."""
    # Core settings
    job_id: str
    book_path: str
    output_format: str = "m4b"
    
    # TTS settings
    engine: str = "orpheus"
    voice: str = "zac"
    dialogue_voice: Optional[str] = None
    use_separate_dialogue: bool = False
    
    # Generation settings
    add_emotion_tags: bool = False
    generate_m4b: bool = True
    use_postprocessing: bool = False
    
    # VibeVoice specific
    reference_audio_path: Optional[str] = None
    vibevoice_temperature: float = 0.7
    vibevoice_temperature: float = 0.7
    vibevoice_top_p: float = 0.95
    
    # Verification
    verification_enabled: bool = False
    verification_model: str = "large-v3" # Use large-v3 by default for best accuracy per user request
    
    # Concurrency (reduced for large models)
    concurrency: int = 4
    
    # Directories (auto-populated from job_id if not set)
    temp_audio_dir: Optional[str] = None
    line_segments_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.temp_audio_dir is None:
            self.temp_audio_dir = job_manager.get_job_dir(self.job_id)
        if self.line_segments_dir is None:
            self.line_segments_dir = job_manager.get_job_line_segments_dir(self.job_id)
        
        # M4B implies we need the book for metadata
        if self.output_format.lower() == "m4b":
            self.generate_m4b = True


class AudiobookPipeline:
    """
    Orchestrates the audiobook generation process.
    
    This class coordinates:
    1. Text loading and preparation
    2. Line-by-line TTS generation with resume support
    3. Chapter assembly
    4. Final audiobook creation
    
    Usage:
        config = PipelineConfig(job_id="abc123", book_path="/path/to/book.epub")
        pipeline = AudiobookPipeline(config)
        async for progress in pipeline.run():
            print(progress)
    
    This is the SINGLE ENTRY POINT for audiobook generation - both API and
    Gradio should use this class.
    """
    
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N lines
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.job_id = config.job_id
        self._stage = PipelineStage.INITIALIZING
        self._lines: List[str] = []
        self._completed_lines: Set[int] = set()
        self._failed_lines: Dict[int, str] = {}  # line_index -> error
        
        # Initialize components
        self._text_loader = TextLoader(config.job_id)
        self._text_loader = TextLoader(config.job_id)
        self._line_generator: Optional[LineGenerator] = None
        self._chapter_assembler: Optional[ChapterAssembler] = None
        self._verifier: Optional[TranscriptionVerifier] = None
        
        if config.verification_enabled:
             # Use CPU/Auto for verification to avoid VRAM contention if using GPU TTS
             self._verifier = TranscriptionVerifier(model_size=config.verification_model)
    
    @property
    def stage(self) -> PipelineStage:
        return self._stage
    
    async def run(self) -> AsyncGenerator[JobProgress, None]:
        """
        Run the complete audiobook generation pipeline.
        
        Yields:
            JobProgress updates throughout the process.
        """
        try:
            # Ensure directories exist
            os.makedirs(self.config.temp_audio_dir, exist_ok=True)
            os.makedirs(self.config.line_segments_dir, exist_ok=True)
            
            # Stage 1: Load and prepare text
            self._stage = PipelineStage.LOADING_TEXT
            yield JobProgress("Loading book text...", 0, self._stage.value)
            
            self._lines = self._text_loader.load(self.config.add_emotion_tags)
            total_lines = len(self._lines)
            logger.info(f"Loaded {total_lines} lines from book")
            
            # Get completed lines from DB for resume
            self._completed_lines = job_manager.get_completed_lines(self.job_id)
            remaining = total_lines - len(self._completed_lines)
            
            if self._completed_lines:
                percent = len(self._completed_lines) / total_lines * 100
                yield JobProgress(
                    f"Resuming: {len(self._completed_lines)} lines done, {remaining} remaining",
                    percent,
                    self._stage.value
                )
            
            # Stage 2: Generate audio for each line
            self._stage = PipelineStage.GENERATING_AUDIO
            yield JobProgress("Starting audio generation (loading model...)", 1, self._stage.value)
            
            # Stage 2: Generate audio for each line
            self._stage = PipelineStage.GENERATING_AUDIO
            
            # Log exact configuration
            logger.info(f"Pipeline Config: Engine={self.config.engine}, Voice={self.config.voice}")
            logger.info(f"Quantization: VibeVoice force_8bit=True (if applicable)")
            logger.info(f"Verification: {'ENABLED' if self.config.verification_enabled else 'DISABLED'}")
            
            yield JobProgress("Starting audio generation (loading model...)", 1, self._stage.value)
            
            async for progress in self._generate_audio():
                yield progress
            
            # Stage 3: Assemble chapters
            self._stage = PipelineStage.ASSEMBLING_CHAPTERS
            yield JobProgress("Assembling chapters...", 90, self._stage.value)
            
            async for progress in self._assemble_chapters():
                yield progress
            
            # Stage 4: Finalize audiobook
            self._stage = PipelineStage.FINALIZING
            yield JobProgress("Creating final audiobook...", 95, self._stage.value)
            
            output_path = await self._finalize()
            
            self._stage = PipelineStage.COMPLETED
            yield JobProgress(f"Audiobook complete: {output_path}", 100, self._stage.value)
            
        except Exception as e:
            self._stage = PipelineStage.FAILED
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def _generate_audio(self) -> AsyncGenerator[JobProgress, None]:
        """Generate audio for each line that hasn't been completed."""
        # Initialize line generator
        tts_kwargs = {}
        if self.config.engine.lower() == "vibevoice":
            tts_kwargs["temperature"] = self.config.vibevoice_temperature
            tts_kwargs["top_p"] = self.config.vibevoice_top_p
            if self.config.reference_audio_path:
                tts_kwargs["reference_audio"] = self.config.reference_audio_path
        
        # Reduce concurrency for large models
        concurrency = self.config.concurrency
        if self.config.engine.lower() == "vibevoice":
            concurrency = 1  # VibeVoice is memory-intensive
        
        self._line_generator = LineGenerator(
            engine=self.config.engine,
            narrator_voice=self.config.voice,
            dialogue_voice=self.config.dialogue_voice,
            use_dialogue_split=self.config.use_separate_dialogue,
            concurrency=concurrency,
            verifier=self._verifier,
            **tts_kwargs
        )
        
        total = len(self._lines)
        completed = len(self._completed_lines)
        last_checkpoint = completed
        
        # Process lines in batches for efficiency and verification
        # Build initial queue
        pending_indices = [i for i in range(total) if i not in self._completed_lines]
        
        # Determine batch size - smaller for VibeVoice to allow regular verification/unload
        # Uses module constant BATCH_SIZE
        
        while pending_indices:
            # Take next batch
            current_batch = pending_indices[:BATCH_SIZE]
            pending_indices = pending_indices[BATCH_SIZE:]
            
            logger.info(f"Processing batch of {len(current_batch)} lines (Start index: {current_batch[0]})...")
            if gpu_manager:
                gpu_manager.log_gpu_stats("Batch Start")
            
            # Yield progress at batch start for UI feedback
            completed = len(self._completed_lines)
            percent = (completed / total) * 100
            yield JobProgress(f"Processing batch ({completed}/{total} lines done)...", percent, self._stage.value)
            
            # --- PHASE 1: GENERATION (TTS) ---
            # Ensure TTS model is loaded
            if self.config.engine.lower() == "vibevoice":
                gpu_manager.acquire_vibevoice()
            elif self.config.engine.lower() == "orpheus":
                gpu_manager.acquire_orpheus()
            
            # Disable inline verification in LineGenerator (we do it in batch)
            if self._line_generator.verifier:
                 self._line_generator.verifier = None

            batch_tasks = []
            for i in current_batch:
                line = self._lines[i]
                output_path = os.path.join(self.config.line_segments_dir, f"line_{i:06d}.wav")
                task = asyncio.create_task(self._process_line(i, line, output_path))
                batch_tasks.append((i, task))
            
            # Process results as they complete (enables per-line progress updates)
            generated_indices = []
            
            for completed_task in asyncio.as_completed([t for _, t in batch_tasks]):
                try:
                    res = await completed_task
                    
                    # Result guaranteed to have index by _process_line
                    idx = res.get("index")
                    if idx is None:
                         logger.error(f"Got result without index: {res}")
                         continue
                    
                    if res.get("success"):
                        generated_indices.append(idx)
                        
                        # Fix for Line Repetition Bug:
                        # If verification is disabled, mark complete IMMEDIATELY to save progress
                        # in case of crash/stall during the batch.
                        if not self.config.verification_enabled:
                            self._completed_lines.add(idx)
                            job_manager.mark_line_complete(self.job_id, idx)

                    elif res.get("skipped"):
                        # Skipped lines are "done", don't verify
                        self._completed_lines.add(idx)
                        job_manager.mark_line_complete(self.job_id, idx)
                    elif res.get("failed"):
                        logger.error(f"Generation failed for line {idx}: {res.get('error')}")
                        self._failed_lines[idx] = res.get("error")
                        
                        # Fail fast if critical mass of errors
                        if len(self._failed_lines) > 5 and len(self._completed_lines) == 0:
                            raise RuntimeError(f"Aborting: First 5 lines failed (Last error: {res.get('error')})")
                        elif len(self._failed_lines) > len(self._lines) * 0.5 and len(self._lines) > 10:
                            raise RuntimeError("Aborting: More than 50% of lines failed")
                    
                    # Update progress after EACH line completes
                    lines_done = len(self._completed_lines) + len(generated_indices)
                    percent = (lines_done / total) * 100
                    
                    # Yield per-line progress to UI
                    yield JobProgress(
                        f"Generating audio... ({lines_done}/{total} lines)",
                        percent,
                        self._stage.value
                    )
                    
                except Exception as e:
                    if "Aborting" in str(e):
                        raise
                    logger.error(f"Unexpected error in batch loop: {e}")
            
            # Yield progress at end of batch for generator consumers
            completed = len(self._completed_lines) + len(generated_indices)
            percent = (completed / total) * 100
            yield JobProgress(f"Batch complete ({completed}/{total} lines)", percent, self._stage.value)

            # Release TTS to free VRAM (always release after batch to prevent VRAM buildup)
            if self.config.engine.lower() == "vibevoice":
                # For VibeVoice, explicitly unload using TTSService
                await tts_service.unload_engine("vibevoice")
                # Also signal GPU manager (legacy/fallback)
                gpu_manager.release_vibevoice()
                
                # Wait for deferred VRAM cleanup to complete
                import gc
                import torch
                for _ in range(3):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                await asyncio.sleep(2)  # Give GPU time to release
                
            elif self.config.engine.lower() == "orpheus":
                gpu_manager.release_orpheus(immediate=True)
            
            # --- PHASE 2: VERIFICATION (if enabled) ---
            # --- PHASE 2: VERIFICATION (if enabled) ---
            if self.config.verification_enabled and generated_indices and self._verifier:
                yield JobProgress(f"Verifying batch of {len(generated_indices)} lines with Whisper...", completed / total * 100, self._stage.value)
                gpu_manager.acquire_whisper()
                
                failed_validation = []
                for idx in generated_indices:
                    line_text = self._lines[idx]
                    audio_path = os.path.join(self.config.line_segments_dir, f"line_{idx:06d}.wav")
                    
                    # Run verification
                    try:
                        passed, score, transcript = await asyncio.to_thread(
                            self._verifier.verify, audio_path, line_text
                        )
                        
                        if not passed:
                            logger.warning(f"Line {idx} failed verification (score {score:.2f}). Retry queued.")
                            failed_validation.append(idx)
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                        else:
                            # Passed! Mark complete
                            self._completed_lines.add(idx)
                            job_manager.mark_line_complete(self.job_id, idx)
                    except Exception as e:
                        logger.error(f"Verification error line {idx}: {e}")
                        # Assume fail on error? Or pass? Let's fail safety.
                        failed_validation.append(idx)
                
                gpu_manager.release_whisper()
                self._verifier.unload() # Ensure memory is freed
                
                # --- PHASE 3: RE-QUEUE ---
                if failed_validation:
                    logger.info(f"Re-queuing {len(failed_validation)} failed lines.")
                    yield JobProgress(f"⚠️ Verification failed for {len(failed_validation)} lines. Re-queuing...", completed / total * 100, self._stage.value)
                    # Add failed lines to FRONT of pending queue to retry immediately
                    pending_indices = failed_validation + pending_indices
            
            # Update Progress
            completed = len(self._completed_lines)
            percent = (completed / total) * 100
            yield JobProgress(f"Generating audio... ({completed}/{total} lines)", percent, self._stage.value)

            
            # Checkpoint save
            if completed - last_checkpoint >= self.CHECKPOINT_INTERVAL:
                last_checkpoint = completed
                checkpoint = JobCheckpoint(
                    total_lines=total,
                    lines_completed=completed,
                    book_file_path=self.config.book_path,
                    add_emotion_tags=self.config.add_emotion_tags,
                    temp_audio_dir=self.config.temp_audio_dir
                )
                job_manager.update_job_checkpoint(self.job_id, checkpoint)
        
        # Report failed lines
        if self._failed_lines:
            failed_count = len(self._failed_lines)
            yield JobProgress(
                f"Completed with {failed_count} failed lines",
                90,
                self._stage.value,
                details={"failed_lines": list(self._failed_lines.keys())}
            )
    
    async def _process_line(self, index: int, line: str, output_path: str) -> Dict[str, Any]:
        """Process a single line with error handling."""
        try:
            return await self._line_generator.generate_line(index, line, output_path)
        except Exception as e:
            logger.error(f"Error processing line {index}: {e}")
            return {"index": index, "failed": True, "error": str(e)}
    
    async def _assemble_chapters(self) -> AsyncGenerator[JobProgress, None]:
        """Assemble line audio files into chapters."""
        self._chapter_assembler = ChapterAssembler(
            temp_audio_dir=self.config.temp_audio_dir,
            line_segments_dir=self.config.line_segments_dir
        )
        
        # Organize lines into chapters
        if not self._completed_lines:
             raise RuntimeError("Job Failed: No audio lines were successfully generated. Cannot assemble audiobook.")
             
        logger.info(f"Finalizing job. Total lines: {len(self._lines)}, Completed lines: {len(self._completed_lines)}")
        logger.info(f"Completed line indices sample: {list(self._completed_lines)[:20]}")
        chapter_files = self._chapter_assembler.organize_chapters(
            self._lines,
            self._completed_lines
        )
        
        yield JobProgress(
            f"Assembling {len(chapter_files)} chapters...",
            91,
            self._stage.value
        )
        
        # Assemble chapter files
        await self._chapter_assembler.assemble_chapters(
            chapter_files,
            use_postprocessing=self.config.use_postprocessing
        )
        
        yield JobProgress(
            f"Assembled {len(chapter_files)} chapters",
            94,
            self._stage.value
        )
    
    async def _finalize(self) -> str:
        """Create the final audiobook file."""
        if not self._chapter_assembler:
            raise RuntimeError("Chapters not assembled")
        
        # Determine book path for metadata
        book_path = self.config.book_path if self.config.generate_m4b else None
        
        output_path = await self._chapter_assembler.finalize(
            book_path=book_path,
            output_format=self.config.output_format.lower().replace("m4b (chapters & cover)", "m4b"),
        )
        
        # Clean up line segments
        self._chapter_assembler.cleanup_line_segments()
        
        return output_path
