
import os
import io
import wave
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pydub import AudioSegment
import traceback

from audiobook.utils.gpu_resource_manager import gpu_manager

class SmartSampleSelector:
    """
    Intelligently selects the best speech segments from audio files.
    Uses Faster-Whisper for segmentation and scoring.
    """
    
    def __init__(self, model_size: str = "medium.en"):
        self.model_size = model_size
        self._model = None
        self._initialized = False

    def _load_model(self):
        """Load Faster-Whisper model on demand."""
        if self._initialized:
            return True
        
        try:
            from faster_whisper import WhisperModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🧠 Loading Whisper model ({self.model_size}) on {device}...")
            
            # fast-whisper handles its own VRAM quite well, but we should clear others
            self._model = WhisperModel(
                self.model_size, 
                device=device,
                compute_type="float16" if device == "cuda" else "int8"
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            traceback.print_exc()
            return False

    def _unload_model(self):
        """Free VRAM."""
        if self._model:
            del self._model
            self._model = None
            self._initialized = False
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process(
        self, 
        audio_path: str, 
        target_duration_sec: float = 90.0,
        output_path: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Process audio to select the best segments up to target duration.
        """
        if not os.path.exists(audio_path):
            return None, "Input file not found"

        # 1. Acquire GPU (stops LLMs)
        success, msg = gpu_manager.acquire_vibevoice() # Reusing the "heavy task" lock
        if not success:
            print(f"⚠️ Could not acquire exclusive GPU access: {msg}")
            # Proceed anyway? Faster-whisper might fit if small enough, but safer to respect manager
        
        try:
            if not self._load_model():
                return None, "Failed to load Whisper model"

            print(f"🎧 Transcribing and segmentation: {os.path.basename(audio_path)}")
            
            # 2. Transcribe & Segment
            # beam_size=5 ensures better accuracy for timestamps
            segments, _ = self._model.transcribe(
                audio_path, 
                beam_size=5, 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert generator to list
            segments = list(segments)
            if not segments:
                return None, "No speech detected"
            
            # 3. Score Segments
            scored_segments = []
            audio = AudioSegment.from_file(audio_path)
            
            for seg in segments:
                duration = seg.end - seg.start
                
                # Filter out very short or very long segments
                if duration < 3.0 or duration > 15.0:
                    continue
                
                # Get audio chunk to calculate energy
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)
                chunk = audio[start_ms:end_ms]
                
                # RMS Energy score (0-1 normalized roughly)
                rms = chunk.rms
                if rms < 100: # Silence threshold
                    continue
                    
                # Combined Score:
                # - Log probability (confidence)
                # - Duration bonus (prefer ~5-10s sentences)
                # - No hallucination (avg_logprob > -1.0)
                
                if seg.avg_logprob < -1.0: # Likely hallucination
                    continue
                
                # Simple score: length * confidence
                score = duration * (1.0 + seg.avg_logprob) 
                
                scored_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "score": score,
                    "duration": duration,
                    "chunk": chunk
                })
            
            # 4. Select Best Segments (Greedy Knapsack)
            # Sort by score descending to pick best content
            scored_segments.sort(key=lambda x: x["score"], reverse=True)
            
            selected = []
            current_duration = 0.0
            
            for seg in scored_segments:
                if current_duration + seg["duration"] <= target_duration_sec:
                    selected.append(seg)
                    current_duration += seg["duration"]
                
                if current_duration >= target_duration_sec * 0.95:
                    break
            
            # 5. Re-order Chronologically for natural flow
            selected.sort(key=lambda x: x["start"])
            
            if not selected:
                return None, "No suitable segments found"
            
            # 6. Concatenate with Crossfade
            print(f"✂️  Concatenating {len(selected)} segments ({current_duration:.1f}s)...")
            final_audio = selected[0]["chunk"]
            
            for i in range(1, len(selected)):
                # 300ms crossfade to smooth transitions
                final_audio = final_audio.append(selected[i]["chunk"], crossfade=300)
            
            # 7. Export
            if output_path is None:
                base, ext = os.path.splitext(audio_path)
                output_path = f"{base}_smart_select.wav"
            
            # Ensure format is WAV 48k mono for training
            final_audio = final_audio.set_frame_rate(48000).set_channels(1)
            final_audio.export(output_path, format="wav")
            
            print(f"✅ Saved smart selection to: {output_path}")
            return output_path, f"Created {current_duration:.1f}s sample from {len(selected)} segments"

        except Exception as e:
            traceback.print_exc()
            return None, f"Error processing selection: {e}"
        finally:
            self._unload_model() # Always clean up

# Singleton
sample_selector = SmartSampleSelector()
