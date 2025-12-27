FROM python:3.12-slim

WORKDIR /app

# Install necessary dependencies, including FFmpeg, Calibre, and OpenGL libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    ffmpeg \
    curl \
    libegl1 \
    libopengl0 \
    libxcb-cursor0 \
    libnss3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Install Calibre
RUN curl -sS https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin

# Make Calibre CLI tools globally accessible
RUN ln -s /opt/calibre/ebook-convert /usr/local/bin/ebook-convert && \
    ln -s /opt/calibre/ebook-meta /usr/local/bin/ebook-meta

# Set PYTHONPATH for both main app and orpheus_tts
ENV PYTHONPATH=/app:/app/orpheus_tts

# Copy requirements files first (for better caching)
COPY requirements.txt .
COPY orpheus_tts/requirements.txt orpheus_tts_requirements.txt

# Install build dependencies first (required for building pkuseg, a chatterbox-tts dependency)
RUN pip install numpy cython setuptools wheel

# Install Python dependencies with no build isolation to use system numpy
RUN uv pip install --system --no-cache-dir --no-build-isolation -r requirements.txt

# Install Orpheus TTS specific dependencies
RUN pip install --no-cache-dir snac==1.2.1 sounddevice==0.4.6 psutil==5.9.0

# Copy the rest of the application files
COPY . .

# Create required directories
RUN mkdir -p /app/generated_audiobooks /app/orpheus_tts/outputs

# Expose ports for Gradio (7860) and Orpheus TTS API (8880)
EXPOSE 7860 8880

# Run the combined server
CMD ["python", "start_server.py"]
