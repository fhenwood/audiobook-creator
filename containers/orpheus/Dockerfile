# Orpheus TTS - Cross-Platform Dockerfile
# Based on Lex-au/Orpheus-FastAPI (Apache 2.0 License)
# Supports: GPU (NVIDIA) on Linux/Windows, Metal on Mac, CPU fallback

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch - pip auto-selects correct version for platform
# On Linux x86_64: installs CUDA-enabled version (uses GPU if available, CPU otherwise)
# On Mac ARM: installs MPS-enabled version (uses Metal if available)
# On Windows: installs CUDA-enabled version
RUN pip install --no-cache-dir torch torchaudio \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p outputs

# Expose port
EXPOSE 8880

# Run the server
CMD ["python", "app.py"]
