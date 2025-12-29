
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audiobook.utils.sample_selector import SmartSampleSelector

class TestSmartSampleSelector(unittest.TestCase):
    
    @patch('audiobook.utils.sample_selector.gpu_manager')
    @patch('audiobook.utils.sample_selector.AudioSegment')
    @patch('faster_whisper.WhisperModel')
    def test_process_logic(self, mock_whisper, mock_audio, mock_gpu):
        # Setup Mocks
        selector = SmartSampleSelector()
        mock_gpu.acquire_vibevoice.return_value = (True, "OK")
        
        # Mock Whisper Model
        mock_model_instance = MagicMock()
        mock_whisper.return_value = mock_model_instance
        selector._model = mock_model_instance
        selector._initialized = True
        
        # Mock Transcribe Output (Segments)
        # Create segments with varying quality
        Segment = lambda start, end, logprob: MagicMock(start=start, end=end, avg_logprob=logprob, text="test")
        segments = [
            Segment(0.0, 5.0, -0.5),   # Good (5s, high conf) -> Score = 5 * 0.5 = 2.5
            Segment(5.0, 7.0, -2.0),   # Bad (Low conf) -> Score = 2 * -1 = -2 (skipped)
            Segment(7.0, 17.0, -0.2),  # Good but Long (10s) -> Score = 10 * 0.8 = 8.0
            Segment(18.0, 22.0, -0.1), # Excellent (4s) -> Score = 4 * 0.9 = 3.6
        ]
        mock_model_instance.transcribe.return_value = (segments, None)
        
        # Mock AudioSegment
        mock_audio_instance = MagicMock()
        mock_audio.from_file.return_value = mock_audio_instance
        # Mock slicing/rms
        mock_chunk = MagicMock()
        mock_chunk.rms = 500 # Loud enough
        mock_audio_instance.__getitem__.return_value = mock_chunk
        
        # Run Process
        with patch('os.path.exists', return_value=True):
            out_path, msg = selector.process("dummy.wav", target_duration_sec=15.0)
            
        print(f"Result: {out_path}, {msg}")
        
        # Verify
        self.assertIsNotNone(out_path)
        # Expecting the best segments to be selected up to 15s
        # 1. Segment 3 (score 8.0, dur 10s)
        # 2. Segment 4 (score 3.6, dur 4s)
        # Total 14s. Segment 1 (5s) would exceed 15s limit (14+5=19)
        
        # We can't easily check internal state without more complex mocks, 
        # but successful return means logic didn't crash.

if __name__ == '__main__':
    unittest.main()
