
import pytest
import asyncio
import os
from unittest.mock import MagicMock, patch, AsyncMock
from audiobook.tts import pipeline as pipeline_module
from audiobook.tts.pipeline import AudiobookPipeline, PipelineConfig, PipelineStage
from audiobook.utils.job_manager import Job, JobStatus

# Mock data
MOCK_TEXT = ["Line 1", "Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]

@pytest.fixture
def mock_gpu_manager():
    with patch.object(pipeline_module, "gpu_manager") as mock:
        yield mock

# Removed mock_text_loader - using real TextLoader reading test.txt

@pytest.fixture
def mock_line_generator():
    with patch.object(pipeline_module, "LineGenerator") as mock_cls:
        generator = mock_cls.return_value
        # Simulate processing - simulate async
        generator.generate_line = AsyncMock(return_value={"success": True, "audio_path": "/tmp/fake.wav"})
        yield mock_cls

@pytest.fixture
def mock_verifier():
    with patch.object(pipeline_module, "TranscriptionVerifier") as mock_cls:
        verifier = mock_cls.return_value
        # Default to pass batch
        async def fake_verify_batch(paths, texts):
            return [(True, 0.95, t) for t in texts]
            
        verifier.verify_batch = AsyncMock(side_effect=fake_verify_batch)
        yield mock_cls

@pytest.fixture
def mock_chapter_assembler():
    with patch.object(pipeline_module, "ChapterAssembler") as mock_cls:
        instance = mock_cls.return_value
        instance.assemble_chapters = AsyncMock(return_value="/tmp/output.m4b")
        instance.finalize = AsyncMock(return_value="/tmp/output.m4b")
        yield mock_cls

@pytest.fixture(autouse=True)
def patch_batch_size():
    with patch.object(pipeline_module, "BATCH_SIZE", 2):
        yield

def test_batch_verification_flow(mock_gpu_manager, mock_line_generator, mock_verifier, mock_chapter_assembler):
    """Test that the pipeline correctly batches generation and verification."""
    async def run():
        config = PipelineConfig(
            job_id="test_batch_job",
            book_path="test.txt",
            engine="orpheus",
            voice="zac",
            verification_enabled=True,
        )
        
        # Using real TextLoader with test.txt
        
        pipeline = AudiobookPipeline(config)
        
        # Run pipeline
        results = []
        async for progress in pipeline.run():
            results.append(progress)
        
        # Assertions
        assert len(results) > 0
        assert pipeline._stage == PipelineStage.COMPLETED
        
        # Check Verification happened
        # 6 lines, batch size 2 -> 3 batches
        assert mock_verifier.return_value.verify_batch.call_count == 3
        
        # Check GPU Management
        assert mock_gpu_manager.acquire_orpheus.called
        assert mock_gpu_manager.acquire_whisper.called
        
    asyncio.run(run())


def test_failed_verification_retry(mock_gpu_manager, mock_line_generator, mock_verifier, mock_chapter_assembler):
    """Test that failed verification triggers a retry."""
    async def run():
        # Using real TextLoader
        
        verifier = mock_verifier.return_value
        
        call_count = 0
        
        async def side_effect(paths, texts):
            nonlocal call_count
            call_count += 1
            # Mock behavior:
            # Batch 1 (Line 1, Line 2) -> Fail Line 2 on first call
            # texts from files: "Line 1", "Line 2".
            if call_count == 1:
                # We expect texts to be correct.
                # Assuming batch size 2, texts should be [L1, L2].
                return [(True, 0.9, texts[0]), (False, 0.4, "Bad transcription")]
            
            # Subsquent batches/retries -> Pass
            return [(True, 0.95, t) for t in texts]
        
        verifier.verify_batch.side_effect = side_effect
        
        config = PipelineConfig(
            job_id="test_retry_job",
            book_path="test.txt",
            verification_enabled=True
        )
        
        pipeline = AudiobookPipeline(config)
        
        async for _ in pipeline.run():
            pass
            
        assert verifier.verify_batch.call_count >= 3

    asyncio.run(run())


def test_job_resume_configuration(mock_gpu_manager, mock_line_generator, mock_chapter_assembler):
    """Test that resume config works (integration style)."""
    async def run():
        # Using real TextLoader
        
        config = PipelineConfig(
            job_id="test_resume_job",
            book_path="test.txt",
            verification_enabled=True
        )
        
        pipeline = AudiobookPipeline(config)
        
        # Simulate partial completion
        pipeline._completed_lines.add(0) # Line 0 done
        pipeline._completed_lines.add(1) # Line 1 done
        
        async for _ in pipeline.run():
            pass
            
        assert mock_line_generator.return_value.generate_line.call_count == 4

    asyncio.run(run())
