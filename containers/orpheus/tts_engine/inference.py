"""
Orpheus TTS - Token Generation and API Handling
Based on Lex-au/Orpheus-FastAPI (Apache 2.0 License)

Handles token generation from the LLM inference server and converts to audio.
"""

import os
import sys
import requests
import json
import time
import wave
import numpy as np
import argparse
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from dotenv import load_dotenv

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

IS_RELOADER = is_reloader_process()
if not IS_RELOADER:
    os.environ['UVICORN_STARTED'] = 'true'

load_dotenv()

# Detect hardware capabilities
import torch
import psutil

HIGH_END_GPU = False
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    gpu_mem_gb = props.total_memory / (1024**3)
    compute_capability = f"{props.major}.{props.minor}"
    
    HIGH_END_GPU = (gpu_mem_gb >= 16.0 or props.major >= 8 or (gpu_mem_gb >= 12.0 and props.major >= 7))
        
    if HIGH_END_GPU:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: High-end CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using high-performance optimizations")
    else:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using GPU-optimized settings")
else:
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if not IS_RELOADER:
        print(f"🖥️ Hardware: CPU only (No CUDA GPU detected)")
        print(f"📊 CPU: {cpu_cores} cores, {cpu_threads} threads")
        print(f"📊 RAM: {ram_gb:.2f} GB")
        print("⚙️ Using CPU-optimized settings")

# API connection settings
API_URL = os.environ.get("ORPHEUS_API_URL")
if not API_URL:
    print("WARNING: ORPHEUS_API_URL not set. API calls will fail until configured.")

HEADERS = {"Content-Type": "application/json"}

try:
    REQUEST_TIMEOUT = int(os.environ.get("ORPHEUS_API_TIMEOUT", "120"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_API_TIMEOUT value, using 120 seconds as fallback")
    REQUEST_TIMEOUT = 120

try:
    MAX_TOKENS = int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192"))
except (ValueError, TypeError):
    MAX_TOKENS = 8192

try:
    TEMPERATURE = float(os.environ.get("ORPHEUS_TEMPERATURE", "0.4"))  # Lower temperature for more consistent voice
except (ValueError, TypeError):
    TEMPERATURE = 0.4

try:
    TOP_P = float(os.environ.get("ORPHEUS_TOP_P", "0.9"))
except (ValueError, TypeError):
    TOP_P = 0.9

# Repetition penalty is hardcoded to 1.1 for stable generation
REPETITION_PENALTY = 1.1

try:
    SAMPLE_RATE = int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000"))
except (ValueError, TypeError):
    SAMPLE_RATE = 24000

if not IS_RELOADER:
    print(f"Configuration loaded:")
    print(f"  API_URL: {API_URL}")
    print(f"  MAX_TOKENS: {MAX_TOKENS}")
    print(f"  TEMPERATURE: {TEMPERATURE}")
    print(f"  TOP_P: {TOP_P}")
    print(f"  REPETITION_PENALTY: {REPETITION_PENALTY}")

NUM_WORKERS = 4 if HIGH_END_GPU else 2

# Define voices by language
ENGLISH_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
FRENCH_VOICES = ["pierre", "amelie", "marie"]
GERMAN_VOICES = ["jana", "thomas", "max"]
KOREAN_VOICES = ["유나", "준서"]
HINDI_VOICES = ["ऋतिका"]
MANDARIN_VOICES = ["长乐", "白芷"]
SPANISH_VOICES = ["javi", "sergio", "maria"]
ITALIAN_VOICES = ["pietro", "giulia", "carlo"]

AVAILABLE_VOICES = (
    ENGLISH_VOICES + 
    FRENCH_VOICES + 
    GERMAN_VOICES + 
    KOREAN_VOICES + 
    HINDI_VOICES + 
    MANDARIN_VOICES + 
    SPANISH_VOICES + 
    ITALIAN_VOICES
)
DEFAULT_VOICE = "tara"

VOICE_TO_LANGUAGE = {}
VOICE_TO_LANGUAGE.update({voice: "english" for voice in ENGLISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "french" for voice in FRENCH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "german" for voice in GERMAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "korean" for voice in KOREAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "hindi" for voice in HINDI_VOICES})
VOICE_TO_LANGUAGE.update({voice: "mandarin" for voice in MANDARIN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "spanish" for voice in SPANISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "italian" for voice in ITALIAN_VOICES})

AVAILABLE_LANGUAGES = ["english", "french", "german", "korean", "hindi", "mandarin", "spanish", "italian"]

# Import from speechpipe
from .speechpipe import turn_token_into_id, CUSTOM_TOKEN_PREFIX

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]


class PerformanceMonitor:
    """Track and report performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = 2.0
        
    def add_tokens(self, count: int = 1) -> None:
        self.token_count += count
        self._check_report()
        
    def add_audio_chunk(self) -> None:
        self.audio_chunks += 1
        self._check_report()
        
    def _check_report(self) -> None:
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
            
    def report(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:
            return
            
        tokens_per_sec = self.token_count / elapsed
        chunks_per_sec = self.audio_chunks / elapsed
        est_duration = self.audio_chunks * 0.085
        
        print(f"Progress: {tokens_per_sec:.1f} tokens/sec, est. {est_duration:.1f}s audio generated")


perf_monitor = PerformanceMonitor()


def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    
    return f"{special_start}{formatted_prompt}{special_end}"


def generate_tokens_from_api(prompt: str, voice: str = DEFAULT_VOICE, temperature: float = TEMPERATURE, 
                             top_p: float = TOP_P, max_tokens: int = MAX_TOKENS, 
                             repetition_penalty: float = REPETITION_PENALTY) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI-compatible API with streaming."""
    start_time = time.time()
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt[:100]}...")
    
    if HIGH_END_GPU:
        print("Using optimized parameters for high-end GPU")
    elif torch.cuda.is_available():
        print("Using optimized parameters for GPU acceleration")
    
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }
    
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
    payload["model"] = model_name
    
    session = requests.Session()
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            response = session.post(
                API_URL, 
                headers=HEADERS, 
                json=payload, 
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return
            
            buffer = ""
            token_counter = 0
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                for token_text in token_chunk.split('>'):
                                    token_text = f'{token_text}>'
                                    token_counter += 1
                                    perf_monitor.add_tokens()
                                    if token_text:
                                        yield token_text
                        except json.JSONDecodeError as e:
                            continue
            
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            return
            
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error to API at {API_URL}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return


def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result


async def tokens_decoder(token_gen) -> Generator[bytes, None, None]:
    """Simplified token decoder with early first-chunk processing for lower latency."""
    buffer = []
    count = 0
    
    first_chunk_processed = False
    min_frames_first = 7
    min_frames_subsequent = 28
    process_every = 7
    
    start_time = time.time()
    last_log_time = start_time
    token_count = 0
    
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            token_count += 1
            
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    print(f"Token processing rate: {token_count/elapsed:.1f} tokens/second")
                last_log_time = current_time
            
            if not first_chunk_processed:
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
            else:
                if count % process_every == 0 and count >= min_frames_subsequent:
                    buffer_to_proc = buffer[-min_frames_subsequent:]
                    
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total: {len(buffer)}")
                    
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples


def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Optimized synchronous wrapper with parallel processing and efficient file I/O."""
    queue_size = 100 if HIGH_END_GPU else 50
    audio_queue = queue.Queue(maxsize=queue_size)
    audio_segments = []
    
    wav_file = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    batch_size = 32 if HIGH_END_GPU else 16
    
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    
    async def async_token_gen():
        batch = []
        for token in syn_token_gen:
            batch.append(token)
            if len(batch) >= batch_size:
                for t in batch:
                    yield t
                batch = []
        for t in batch:
            yield t
    
    async def async_producer():
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        try:
            producer_started_event.set()
            
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    current_time = time.time()
                    if current_time - last_log_time >= 3.0:
                        elapsed = current_time - last_log_time
                        if elapsed > 0:
                            print(f"Audio generation rate: {chunk_count/elapsed:.2f} chunks/second")
                        last_log_time = current_time
                        chunk_count = 0
        except Exception as e:
            print(f"Error in token processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            print("Producer completed")
            producer_done_event.set()
            audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async, name="TokenProcessor")
    thread.daemon = True
    thread.start()
    
    producer_started_event.wait(timeout=5.0)
    
    write_buffer = bytearray()
    buffer_max_size = 1024 * 1024
    last_check_time = time.time()
    check_interval = 1.0
    
    while True:
        try:
            audio = audio_queue.get(timeout=0.1)
            
            if audio is None:
                print("Received end-of-stream marker")
                break
            
            audio_segments.append(audio)
            
            if wav_file:
                write_buffer.extend(audio)
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()
        
        except queue.Empty:
            current_time = time.time()
            
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                
                if producer_done_event.is_set() and audio_queue.empty():
                    print("Producer done and queue empty")
                    break
                
                if wav_file and len(write_buffer) > 0:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()
    
    if thread.is_alive():
        print("Waiting for token processor thread...")
        thread.join(timeout=10.0)
    
    if wav_file and len(write_buffer) > 0:
        wav_file.writeframes(write_buffer)
    
    if wav_file:
        wav_file.close()
        if output_file:
            print(f"Audio saved to {output_file}")
    
    if audio_segments:
        total_bytes = sum(len(segment) for segment in audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)
        total_time = time.time() - perf_monitor.start_time
        realtime_factor = duration / total_time if total_time > 0 else 0
        
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        
        if realtime_factor < 1.0:
            print("⚠️ Warning: Generation is slower than realtime")
        else:
            print(f"✓ Generation is {realtime_factor:.1f}x faster than realtime")
    
    return audio_segments


def split_text_into_sentences(text):
    """Split text into sentences."""
    parts = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        
        if char in (' ', '\n', '\t') and len(current_sentence) > 1:
            prev_char = current_sentence[-2]
            if prev_char in ('.', '!', '?'):
                if len(current_sentence) > 3 and current_sentence[-3] not in ('.', ' '):
                    parts.append(current_sentence.strip())
                    current_sentence = ""
    
    if current_sentence.strip():
        parts.append(current_sentence.strip())
    
    min_chars = 20
    combined_sentences = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        
        while i < len(parts) - 1 and len(current) < min_chars:
            i += 1
            current += " " + parts[i]
            
        combined_sentences.append(current)
        i += 1
    
    return combined_sentences


def stitch_wav_files(input_files, output_file, crossfade_ms=50):
    """Stitch multiple WAV files together with crossfading."""
    if not input_files:
        return
        
    print(f"Stitching {len(input_files)} WAV files with {crossfade_ms}ms crossfade")
    
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        return
    
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    final_audio = np.array([], dtype=np.int16)
    first_params = None
    
    for i, input_file in enumerate(input_files):
        try:
            with wave.open(input_file, 'rb') as wav:
                if first_params is None:
                    first_params = wav.getparams()
                
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                
                if i == 0:
                    final_audio = audio
                else:
                    if len(final_audio) >= crossfade_samples and len(audio) >= crossfade_samples:
                        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                        
                        crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                                            audio[:crossfade_samples] * fade_in).astype(np.int16)
                        
                        final_audio = np.concatenate([final_audio[:-crossfade_samples], 
                                                     crossfade_region, 
                                                     audio[crossfade_samples:]])
                    else:
                        final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            if i == 0:
                raise
    
    try:
        with wave.open(output_file, 'wb') as output_wav:
            if first_params is None:
                raise ValueError("No valid WAV files were processed")
            output_wav.setparams(first_params)
            output_wav.writeframes(final_audio.tobytes())
        print(f"Successfully stitched audio to {output_file}")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")
        raise


def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None,
                             temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                             repetition_penalty=None, use_batching=True, max_batch_chars=1000):
    """Generate speech from text using Orpheus model."""
    print(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"Using voice: {voice}, GPU: {'Yes (High-end)' if HIGH_END_GPU else 'Yes' if torch.cuda.is_available() else 'No'}")
    
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    if not use_batching or len(prompt) < max_batch_chars:
        result = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt, 
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=output_file
        )
        
        end_time = time.time()
        print(f"Total speech generation completed in {end_time - start_time:.2f} seconds")
        return result
    
    print(f"Using sentence-based batching for text with {len(prompt)} characters")
    
    sentences = split_text_into_sentences(prompt)
    print(f"Split text into {len(sentences)} segments")
    
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            if current_batch:
                current_batch += " "
            current_batch += sentence
    
    if current_batch:
        batches.append(current_batch)
    
    print(f"Created {len(batches)} batches for processing")
    
    all_audio_segments = []
    batch_temp_files = []
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
        
        temp_output_file = None
        if output_file:
            temp_output_file = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
            batch_temp_files.append(temp_output_file)
        
        batch_segments = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=batch,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=temp_output_file
        )
        
        all_audio_segments.extend(batch_segments)
    
    if output_file and batch_temp_files:
        stitch_wav_files(batch_temp_files, output_file)
        
        for temp_file in batch_temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if all_audio_segments:
        total_bytes = sum(len(segment) for segment in all_audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)
        print(f"Generated {len(all_audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {duration/total_time:.2f}x")
        
    print(f"Total speech generation completed in {total_time:.2f} seconds")
    
    return all_audio_segments


def list_available_voices():
    """List all available voices."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")


def main():
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    prompt = args.text
    if not prompt:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--voice", "--output", "--temperature", "--top_p"):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
    
    output_file = args.output
    if not output_file:
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")


if __name__ == "__main__":
    main()
