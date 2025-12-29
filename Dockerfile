# Universal Dockerfile for Audiobook Creator
# Supports Orpheus (CPU/GPU), Chatterbox (CPU/GPU), and VibeVoice (GPU only)
# Base image provides CUDA toolkit for building flash-attention

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH=/app:/app/orpheus_tts

# Create working directory
WORKDIR /app

# Install system dependencies
# python3.11 is recommended for VibeVoice
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    curl \
    build-essential \
    ninja-build \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    libnss3 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv package manager
RUN pip3 install uv

# Install Calibre
RUN curl -sS https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin

# Make Calibre CLI tools globally accessible
RUN ln -s /opt/calibre/ebook-convert /usr/local/bin/ebook-convert && \
    ln -s /opt/calibre/ebook-meta /usr/local/bin/ebook-meta

# Install PyTorch with CUDA support
# This is required for VibeVoice and flash-attention
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install build dependencies for flash-attention
RUN pip3 install --no-cache-dir packaging ninja setuptools wheel

# Install flash-attention (required for VibeVoice)
# Check availability of GPU during build is not possible, so we force build
# Note: This step takes time
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation



# Copy requirements files
COPY requirements.txt .
COPY orpheus_tts/requirements.txt orpheus_tts_requirements.txt

# Install other Python dependencies
# Use --system to install into system python
RUN uv pip install --system --no-cache-dir --no-build-isolation -r requirements.txt

# Install Orpheus dependencies
RUN pip3 install --no-cache-dir snac==1.2.1 sounddevice==0.4.6 psutil==5.9.0

# Install Chatterbox-TTS without dependencies to avoid conflict with VibeVoice's transformers version
RUN pip3 install --no-deps chatterbox-tts

# Install VibeVoice from community repo (Last to ensure dependencies are met)
RUN pip3 install --no-cache-dir git+https://github.com/vibevoice-community/VibeVoice.git

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/generated_audiobooks /app/orpheus_tts/outputs

# Expose ports
EXPOSE 7860 8880

# Run the server
CMD ["python", "start_server.py"]
