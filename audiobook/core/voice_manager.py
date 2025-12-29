import os
import shutil
import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceInfo(BaseModel):
    id: str
    name: str
    path: str
    preview_path: Optional[str] = None

class VoiceManager:
    def __init__(self, voices_dir: str = "static_files/voices", preset_voices_dir: str = "static_files/preset_voices"):
        self.voices_dir = voices_dir
        self.preset_voices_dir = preset_voices_dir
        os.makedirs(self.voices_dir, exist_ok=True)
        
    def list_voices(self) -> List[VoiceInfo]:
        """List all custom and preset voices."""
        voices = []
        
        # Scan user voices directory
        if os.path.exists(self.voices_dir):
            for filename in os.listdir(self.voices_dir):
                if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    path = os.path.abspath(os.path.join(self.voices_dir, filename))
                    name = os.path.splitext(filename)[0].replace("_", " ").title()
                    voices.append(VoiceInfo(id=filename, name=name, path=path))
        
        # Scan preset voices directory (built-in voices)
        if os.path.exists(self.preset_voices_dir):
            for filename in os.listdir(self.preset_voices_dir):
                if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    path = os.path.abspath(os.path.join(self.preset_voices_dir, filename))
                    # Mark preset voices with (Preset) suffix
                    base_name = os.path.splitext(filename)[0].replace("-", " - ").replace("_", " ").title()
                    name = f"{base_name} (Preset)"
                    voices.append(VoiceInfo(id=filename, name=name, path=path))
        
        return sorted(voices, key=lambda x: x.name)

    async def add_voice(
        self, 
        name: str, 
        file_path: str, 
        preprocess: bool = False,
        smart_selection: bool = False,
        target_duration: float = 90.0
    ) -> Optional[Tuple[VoiceInfo, str]]:
        """
        Add a new voice from a file, optionally processing it.
        Returns Tuple[VoiceInfo, status_message] or None
        """
        if not name or not file_path or not os.path.exists(file_path):
            return None
            
        # Sanitize filename
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        # Force WAV extension if processing is done, otherwise preserve
        ext = ".wav" if (preprocess or smart_selection) else os.path.splitext(file_path)[1]
        filename = f"{safe_name}{ext}"
        target_path = os.path.join(self.voices_dir, filename)
        
        status_msg = f"Imported {name}"
        current_file = file_path
        
        try:
            # 1. Preprocessing (Demucs -> Enhancement -> SR)
            if preprocess:
                logger.info(f"Adding voice {name}: Running Preprocessing Pipeline...")
                from audiobook.utils.audio_enhancer import audio_pipeline
                
                # Use a temp path for intermediate step
                temp_processed = os.path.join(self.voices_dir, f"temp_proc_{filename}")
                processed_path, proc_msg = audio_pipeline.process(
                    current_file, 
                    output_path=temp_processed,
                    enable_preprocessing=True
                )
                
                if processed_path:
                    current_file = processed_path
                    status_msg += " + Enhanced"
                else:
                    logger.warning(f"Preprocessing failed: {proc_msg}")
                    status_msg += " (Enhancement Skipped)"

            # 2. Smart Selection (Whisper Segmentation)
            if smart_selection:
                logger.info(f"Adding voice {name}: Running Smart Selection...")
                from audiobook.utils.sample_selector import sample_selector
                
                # Determine output path (if final step, use target_path, else temp)
                sel_out = target_path if not preprocess else target_path # Actually we just overwrite target
                
                selection_path, sel_msg = sample_selector.process(
                    current_file,
                    target_duration_sec=target_duration,
                    output_path=target_path
                )
                
                if selection_path:
                    # Success - file is already at target_path
                    status_msg += f" + Smart Selected ({sel_msg})"
                    # Clean up temp if we created one in step 1
                    if preprocess and os.path.exists(current_file) and current_file != file_path and current_file != target_path:
                         os.remove(current_file)
                    return VoiceInfo(id=filename, name=name, path=target_path), status_msg
                else:
                    logger.warning(f"Smart selection failed: {sel_msg}")
                    status_msg += " (Selection Skipped)"
            
            # 3. Final Copy (if not already saved by selection)
            if current_file != target_path:
                shutil.copy2(current_file, target_path)
                
            # Cleanup intermediate files
            if preprocess and os.path.exists(current_file) and current_file != file_path and current_file != target_path:
                os.remove(current_file)

            return VoiceInfo(id=filename, name=name, path=target_path), status_msg
            
        except Exception as e:
            logger.error(f"Failed to add voice {name}: {e}")
            return None
            
    def delete_voice(self, filename: str) -> bool:
        """Delete a custom voice."""
        target_path = os.path.join(self.voices_dir, filename)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                return True
            except Exception as e:
                logger.error(f"Failed to delete voice {filename}: {e}")
                return False
        return False

# Global instance
voice_manager = VoiceManager()
