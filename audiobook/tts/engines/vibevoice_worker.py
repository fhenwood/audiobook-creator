"""
VibeVoice Worker Process
Runs VibeVoice in a separate process to ensure full VRAM cleanup on exit.
"""
import os
import sys
import time
import queue
import traceback
import multiprocessing as mp
import numpy as np
import torch
import io
import wave
from typing import Any
from dataclasses import dataclass

@dataclass
class WorkerCommand:
    type: str  # INIT, GENERATE, SHUTDOWN
    payload: dict = None

@dataclass
class WorkerResult:
    success: bool
    data: Any = None
    error: str = None

class VibeVoiceWorker:
    def __init__(self, command_queue: mp.Queue, result_queue: mp.Queue, model_size: str = "7b"):
        self.cmd_q = command_queue
        self.res_q = result_queue
        self.model_size = model_size
        self._model = None
        self._processor = None
        self._initialized = False
        
    def run(self):
        """Main worker loop."""
        print(f"[Worker] Started PID: {os.getpid()}")
        
        while True:
            try:
                cmd = self.cmd_q.get()
                
                if cmd.type == "SHUTDOWN":
                    print("[Worker] Received SHUTDOWN")
                    self._shutdown()
                    break
                    
                elif cmd.type == "INIT":
                    self._handle_init()
                    
                elif cmd.type == "GENERATE":
                    self._handle_generate(cmd.payload)
                    
            except Exception as e:
                print(f"[Worker] Critical Error: {e}")
                traceback.print_exc()
                self.res_q.put(WorkerResult(False, error=str(e)))

    def _shutdown(self):
        # We don't need fancy cleanup here because the process is about to exit!
        # That's the whole point of this architecture.
        self._model = None
        self._processor = None
        print("[Worker] Exiting...")
    
    def _handle_init(self):
        try:
            print(f"[Worker] Initializing VibeVoice {self.model_size}...")
            
            # 1. Imports within process
            from audiobook.models.manager import model_manager
            
            # 2. Check dependencies
            try:
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference as VibeVoiceForConditionalGeneration
            except ImportError:
                from vibevoice import VibeVoice as VibeVoiceForConditionalGeneration

            # 3. Model Path
            model_id = f"vibevoice-{self.model_size}"
            model_path = model_manager.get_model_path(model_id)
            load_path = model_path if model_path else "vibevoice/VibeVoice-7B"
            
            # 4. Load Processor
            from transformers import AutoProcessor
            try:
                self._processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
            except:
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
                self._processor = VibeVoiceProcessor.from_pretrained(load_path, trust_remote_code=True)
                
            # 5. Load Model (Force GPU 0)
            print("[Worker] Loading model to GPU 0 (FP16)...")
            self._model = VibeVoiceForConditionalGeneration.from_pretrained(
                load_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map={"": 0}
            )
            
            self._initialized = True
            self.res_q.put(WorkerResult(True, data="Initialized"))
            print("[Worker] Ready.")
            
        except Exception as e:
            print(f"[Worker] Init failed: {e}")
            self.res_q.put(WorkerResult(False, error=str(e)))

    def _handle_generate(self, payload):
        if not self._initialized:
            self.res_q.put(WorkerResult(False, error="Worker not initialized"))
            return

        try:
            text = payload["text"]
            voice = payload.get("voice", "speaker_0")
            temperature = payload.get("temperature", 0.7)
            top_p = payload.get("top_p", 0.95)
            
            # --- Logic Copied from vibevoice.py ---
            
            # 1. Prepare Inputs
            speaker_id = 0
            if isinstance(voice, str) and voice.startswith("speaker_"):
                try:
                    speaker_id = int(voice.split("_")[1])
                except ValueError:
                    pass
            
            script_text = f"Speaker {speaker_id}: {text}"
            
            # 2. Processor
            # Handle reference audio (cloned voice)
            temp_ref_file = None
            reference_audio = payload.get("reference_audio")
            
            try:
                if reference_audio:
                    if isinstance(reference_audio, str) and os.path.exists(reference_audio):
                        # It's a path! Use directly.
                        print(f"[Worker] Using reference audio path: {reference_audio}")
                        inputs = self._processor(text=script_text, voice_samples=[reference_audio], return_tensors="pt")
                    else:
                        # It's bytes (or invalid path), try to write to temp file
                        import tempfile
                        # Create temp file for reference audio
                        temp_ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        if isinstance(reference_audio, str):
                            # Ensure we don't try to write path string as bytes
                            print(f"[Worker] Warning: reference_audio is a string but not a valid path? '{reference_audio[:50]}...'")
                            # If it's a string but doesn't exist, we can't do much. 
                            # But maybe it's base64? Unlikely given pipeline. 
                            # Let's assume bytes only for this branch.
                            raise ValueError(f"reference_audio path not found: {reference_audio}")
                        
                        temp_ref_file.write(reference_audio)
                        temp_ref_file.close()
                        
                        print(f"[Worker] Using reference audio bytes (temp): {temp_ref_file.name}")
                        inputs = self._processor(text=script_text, voice_samples=[temp_ref_file.name], return_tensors="pt")
                    

                else:
                    # Fallback strategies
                    default_voice_paths = [
                        "/app/models/tts/xtts-v2/samples/en_sample.wav",
                        "/app/static_files/voice_samples/zac_sample.wav",
                        "/app/audio_samples/default_voice.wav"
                    ]
                    
                    fallback_voice = None
                    for path in default_voice_paths:
                        if os.path.exists(path):
                            fallback_voice = path
                            break
                    
                    if fallback_voice:
                        inputs = self._processor(text=script_text, voice_samples=[fallback_voice], return_tensors="pt")
                    else:
                        inputs = self._processor(text=script_text, return_tensors="pt")
            finally:
                if temp_ref_file and os.path.exists(temp_ref_file.name):
                    try:
                        os.unlink(temp_ref_file.name)
                    except:
                        pass
                
            # 3. Move to Cuda
            cuda_inputs = {}
            for k, v in inputs.items():
                if v is not None and hasattr(v, "to"):
                    cuda_inputs[k] = v.to("cuda")
            
            # 4. Generate
            audio_output = self._model.generate(
                **cuda_inputs,
                tokenizer=self._processor.tokenizer,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                cfg_scale=1.3
            )
            
            # 5. Extract Audio
            audio_array = None
            if hasattr(audio_output, "speech_outputs") and audio_output.speech_outputs:
                speech_list = audio_output.speech_outputs
                audio_tensors = [s.cpu() for s in speech_list]
                audio_array = torch.cat(audio_tensors, dim=-1).numpy()
            elif hasattr(audio_output, "cpu"):
                 audio_array = audio_output.cpu().numpy()
            else:
                 audio_array = audio_output[0].cpu().numpy()
                 
            # Squeeze
            while len(audio_array.shape) > 1 and audio_array.shape[0] == 1:
                audio_array = audio_array.squeeze(0)
                
            # Convert to int16 bytes
            if audio_array.dtype.kind == 'f':
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
                
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_array.tobytes())
            
            self.res_q.put(WorkerResult(True, data=wav_buffer.getvalue()))
            
        except Exception as e:
            print(f"[Worker] Generation failed: {e}")
            traceback.print_exc()
            self.res_q.put(WorkerResult(False, error=str(e)))

def run_worker(cmd_q, res_q, model_size):
    """Entry point for subprocess."""
    # Ensure standard streams are flushed
    sys.stdout.reconfigure(line_buffering=True)
    worker = VibeVoiceWorker(cmd_q, res_q, model_size)
    worker.run()
