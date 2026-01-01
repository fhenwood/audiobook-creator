import os
import gc
import tempfile
import shutil
import subprocess
import torch
import traceback
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from contextlib import contextmanager


class AudioPreprocessingPipeline:
    """
    Multi-stage audio preprocessing pipeline with explicit GPU memory management.
    
    Stages:
    1. Demucs v4 (htdemucs_ft) - Vocal isolation
    2. ClearerVoice MossFormer2_SE_48K - Speech enhancement
    3. MossFormer2_SR_48K - Super-resolution to 48kHz
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._demucs_available = None
        self._clearvoice_model = None
        self._clearvoice_sr_model = None
        self._use_cuda = torch.cuda.is_available()
    
    def _check_demucs(self) -> bool:
        """Check if Demucs is available."""
        if self._demucs_available is not None:
            return self._demucs_available
        
        try:
            # Demucs is usually run as an external CLI process via python -m demucs
            result = subprocess.run(
                ["python", "-m", "demucs", "--help"],
                capture_output=True,
                timeout=5
            )
            self._demucs_available = result.returncode == 0
        except Exception:
            self._demucs_available = False
        
        return self._demucs_available

    def _unload_models(self):
        """Force unload models and clear VRAM."""
        print("🧹 Purging preprocessing models from VRAM...")
        self._clearvoice_model = None
        self._clearvoice_sr_model = None
        
        # Standard torch cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✅ VRAM cleared")

    @contextmanager
    def gpu_context(self):
        """
        Ensures other GPU services are stopped and models are purged when done.
        """
        try:
            from audiobook.utils.gpu_resource_manager import gpu_manager
            # VibeVoice/Preprocessing both need ~24GB, so we must stop LLM containers
            gpu_manager.stop_llm_services()
            yield
        finally:
            self._unload_models()

    def _load_enhancement(self):
        """Load ClearerVoice enhancement model."""
        if self._clearvoice_model is not None:
            return True
        
        try:
            from clearvoice import ClearVoice
            print("🔧 Loading ClearerVoice SE model (MossFormer2_SE_48K)...")
            self._clearvoice_model = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
            return True
        except Exception as e:
            print(f"⚠️ Failed to load ClearVoice SE: {e}")
            return False

    def _load_sr(self):
        """Load ClearerVoice SR model."""
        if self._clearvoice_sr_model is not None:
            return True
        
        try:
            from clearvoice import ClearVoice
            print("🔧 Loading ClearerVoice SR model (MossFormer2_SR_48K)...")
            self._clearvoice_sr_model = ClearVoice(
                task='speech_super_resolution',
                model_names=['MossFormer2_SR_48K']
            )
            return True
        except Exception as e:
            print(f"⚠️ Failed to load ClearVoice SR: {e}")
            return False

    def _run_demucs(self, input_path: str, output_dir: str) -> Optional[str]:
        """Run Demucs isolation stage."""
        if not self._check_demucs():
            return None
        
        try:
            print("🎵 Stage 1: Demucs Vocal Isolation...")
            # Demucs handles its own GPU allocation
            cmd = [
                "python", "-m", "demucs",
                "--two-stems", "vocals",
                "-n", "htdemucs_ft",
                "-o", output_dir,
                input_path
            ]
            if not self._use_cuda:
                cmd.append("-d")
                cmd.append("cpu")
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"⚠️ Demucs failed: {result.stderr}")
                return None
            
            input_name = Path(input_path).stem
            vocals_path = os.path.join(output_dir, "htdemucs_ft", input_name, "vocals.wav")
            
            if os.path.exists(vocals_path):
                return vocals_path
        except Exception as e:
            print(f"⚠️ Demucs error: {e}")
        return None

    def _run_enhancement(self, input_path: str, output_path: str) -> Optional[str]:
        """Run ClearerVoice SE stage."""
        if not self._load_enhancement():
            return None
        try:
            print("🎤 Stage 2: ClearerVoice Enhancement...")
            enhanced = self._clearvoice_model(input_path=input_path, online_write=False)
            self._clearvoice_model.write(enhanced, output_path=output_path)
            return output_path
        except Exception as e:
            print(f"⚠️ SE failed: {e}")
            return None

    def _run_sr(self, input_path: str, output_path: str) -> Optional[str]:
        """Run ClearerVoice SR stage."""
        if not self._load_sr():
            return None
        try:
            print("🔊 Stage 3: Super-Resolution (48kHz)...")
            sr_audio = self._clearvoice_sr_model(input_path=input_path, online_write=False)
            self._clearvoice_sr_model.write(sr_audio, output_path=output_path)
            return output_path
        except Exception as e:
            print(f"⚠️ SR failed: {e}")
            return None

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        enable_preprocessing: bool = True,
        stages: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Process audio with on-demand GPU loading and forced unloading.
        
        Args:
            input_path: Input file
            output_path: Output file
            enable_preprocessing: Main toggle
            stages: Optional list of stages to run ('demucs', 'enhancement', 'sr')
        """
        if not os.path.exists(input_path):
            return "", "Input file not found"
        
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_enhanced.wav"

        # Default stages if not specified
        if stages is None:
            stages = ['demucs', 'enhancement', 'sr']

        if not enable_preprocessing:
            # Just ensure it's a valid WAV at 48k for consistency
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_path,
                    "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
                    output_path
                ], capture_output=True, timeout=30)
                return output_path, "Converted to 48k WAV"
            except Exception as e:
                print(f"⚠️ Failed to convert audio: {e}")
                shutil.copy(input_path, output_path)
                return output_path, "Saved as is"

        # Run pipeline within GPU context
        with self.gpu_context():
            temp_dir = tempfile.mkdtemp(prefix="enhancer_")
            try:
                current_audio = input_path
                completed = []
                
                # Demucs
                if 'demucs' in stages:
                    v_path = self._run_demucs(current_audio, temp_dir)
                    if v_path:
                        current_audio = v_path
                        completed.append("Demucs")
                
                # Enhancement
                if 'enhancement' in stages:
                    e_path = os.path.join(temp_dir, "se.wav")
                    if self._run_enhancement(current_audio, e_path):
                        current_audio = e_path
                        completed.append("ClearerVoice")
                
                # SR
                if 'sr' in stages:
                    s_path = os.path.join(temp_dir, "sr.wav")
                    if self._run_sr(current_audio, s_path):
                        current_audio = s_path
                        completed.append("Super-Res")
                
                shutil.copy(current_audio, output_path)
                return output_path, f"Enhanced: {', '.join(completed)}"
            except Exception as e:
                traceback.print_exc()
                return input_path, f"Error: {str(e)}"
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def is_fully_available(self) -> Dict[str, bool]:
        """Check component status."""
        return {
            "demucs": self._check_demucs(),
            "cuda": self._use_cuda
        }


# Legacy compatibility and Singleton
audio_pipeline = AudioPreprocessingPipeline()

