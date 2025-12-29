"""
Orpheus TTS - Speech Pipeline
Based on Lex-au/Orpheus-FastAPI (Apache 2.0 License)

Converts token sequences to audio using the SNAC model.
"""

from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

IS_RELOADER = is_reloader_process()
if not IS_RELOADER:
    os.environ['UVICORN_STARTED'] = 'true'

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
        if not IS_RELOADER:
            print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        if not IS_RELOADER:
            print("CUDA graphs support is available")
except:
    pass

# Load SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not IS_RELOADER:
    print(f"Using device: {snac_device}")
model = model.to(snac_device)

if not IS_RELOADER:
    print("Using standard PyTorch optimizations")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
    
    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    
    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    
    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)
        
        # Extract the relevant slice and efficiently convert to bytes
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process on GPU if possible, with minimal data transfer
        if snac_device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
        
        return audio_bytes


# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000


def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
    # Check cache first
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
        
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
        
    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
        
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
            
        return token_id
    except (ValueError, IndexError):
        return None


async def tokens_decoder(token_gen):
    """Optimized token decoder with early first-chunk processing for lower latency"""
    buffer = []
    count = 0
    
    first_chunk_processed = False
    min_frames_first = 7  # Just one chunk for first audio - ultra-low latency
    min_frames_subsequent = 28  # Standard minimum (4 chunks of 7 tokens)
    ideal_frames = 49  # Ideal standard frame size (7×7 window)
    process_every_n = 7  # Process every 7 tokens
    
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    
    async for token_sim in token_gen:
        token_count += 1
        
        token = turn_token_into_id(token_sim, count)
        
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    tokens_per_sec = token_count / elapsed
                    print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_count = 0
            
            # Different processing logic based on whether first chunk has been processed
            if not first_chunk_processed:
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens for low latency")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
            else:
                if count % process_every_n == 0:
                    if len(buffer) >= ideal_frames:
                        buffer_to_proc = buffer[-ideal_frames:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue
                    
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total: {len(buffer)}")
                    
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
    
    # End-of-generation handling - process remaining frames
    if len(buffer) >= ideal_frames:
        buffer_to_proc = buffer[-ideal_frames:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    elif len(buffer) >= min_frames_subsequent:
        buffer_to_proc = buffer[-min_frames_subsequent:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    elif len(buffer) >= process_every_n:
        # Pad with copies of the final token
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding
        print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} padding")
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples


def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with larger queue and parallel processing"""
    max_queue_size = 32 if snac_device == "cuda" else 8
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    batch_size = 16 if snac_device == "cuda" else 4
    
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            if len(token_batch) >= batch_size:
                for t in token_batch:
                    yield t
                token_batch = []
        for t in token_batch:
            yield t

    async def async_producer():
        start_time = time.time()
        chunk_count = 0
        
        try:
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.2f} chunks/sec)")
        except Exception as e:
            print(f"Error in audio producer: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Audio producer completed")
            audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.daemon = True
    thread.start()

    buffer_size = 5
    audio_buffer = []
    
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_buffer.append(audio)
        if len(audio_buffer) >= buffer_size:
            for chunk in audio_buffer:
                yield chunk
            audio_buffer = []
    
    for chunk in audio_buffer:
        yield chunk

    thread.join()
