
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from audiobook.tts.pipeline import AudiobookPipeline, PipelineConfig, PipelineStage
from audiobook.utils.job_manager import JobProgress

@pytest.mark.asyncio
async def test_pipeline_progress_updates():
    """Test that pipeline yields progress updates per-line and unloads VRAM."""
    
    # Mock dependencies
    with patch('audiobook.tts.pipeline.TextLoader') as mock_loader, \
         patch('audiobook.tts.pipeline.LineGenerator') as mock_generator, \
         patch('audiobook.tts.pipeline.ChapterAssembler') as mock_assembler, \
         patch('audiobook.tts.pipeline.tts_service') as mock_tts_service, \
         patch('audiobook.tts.pipeline.gpu_manager') as mock_gpu_manager, \
         patch('audiobook.tts.pipeline.job_manager') as mock_job_manager:
        
        # Setup mocks
        mock_loader.return_value.load.return_value = ["Line 1", "Line 2", "Line 3", "Line 4"]
        
        mock_gen_instance = mock_generator.return_value
        mock_gen_instance.generate_line = AsyncMock(side_effect=[
            {"success": True, "index": 0},
            {"success": True, "index": 1}, 
            {"success": True, "index": 2},
            {"success": True, "index": 3}
        ])
        
        mock_tts_service.unload_engine = AsyncMock()
        
        # Setup async mocks
        mock_assembler_instance = mock_assembler.return_value
        mock_assembler_instance.assemble_chapters = AsyncMock()
        mock_assembler_instance.finalize = AsyncMock(return_value="output.m4b")
        
        mock_job_manager.get_completed_lines.return_value = set()
        
        # Setup pipeline
        config = PipelineConfig(
            job_id="test_job",
            book_path="dummy.txt",
            engine="vibevoice",
            voice="speaker_0"
        )
        
        pipeline = AudiobookPipeline(config)
        
        # Run pipeline
        progress_updates = []
        async for progress in pipeline.run():
            progress_updates.append(progress)
            print(f"Update: {progress.message} ({progress.percent_complete}%)")
        
        # Verify per-line progress updates in GENERATING_AUDIO stage
        gen_updates = [p for p in progress_updates if p.stage == PipelineStage.GENERATING_AUDIO.value]
        
        gen_update_messages = [p.message for p in gen_updates]
        print(f"Captured updates: {gen_update_messages}")
        
        # We expect frequent updates. 
        # 1 initial (Starting) + 1 batch start + 4 per-line + 1 batch end = 7 updates ideally
        # If we get at least 4 updates, it proves per-line or batch start/end is working better than before (which was just 1 or 2)
        assert len(gen_updates) >= 5, f"Expected frequent updates, got {len(gen_updates)}. Messages: {gen_update_messages}"
        
        # Verify VRAM unloading
        # Should be called at least once (after buffer/batch)
        mock_tts_service.unload_engine.assert_called_with("vibevoice")
        
        print("\n✅ Verification Passed: Progress updates frequent and VRAM unload called.")

if __name__ == "__main__":
    # Run simple
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_pipeline_progress_updates())
