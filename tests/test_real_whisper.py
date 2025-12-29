
import os
import sys
from faster_whisper import WhisperModel
import torch

try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    print("Loading Faster-Whisper Model (tiny)...")
    model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
    print("Model loaded successfully.")

    # Create dummy audio
    from pydub import AudioSegment
    from pydub.generators import Sine
    audio = Sine(440).to_audio_segment(duration=3000)
    audio.export("test_whisper.wav", format="wav")
    print("Created test_whisper.wav")

    print("Transcribing...")
    segments, info = model.transcribe("test_whisper.wav", beam_size=5)
    
    count = 0
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        count += 1
    
    print(f"Transcription complete. Found {count} segments.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
