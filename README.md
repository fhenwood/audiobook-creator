# 🎧 Audiobook Creator

Convert ebooks (EPUB, PDF, TXT) into high-quality audiobooks with expressive AI voices.

**Supports two TTS engines:**
- **Orpheus TTS** - 8 built-in voices with emotion tags (`<laugh>`, `<sigh>`, etc.)
- **Chatterbox TTS** - Zero-shot voice cloning from any audio sample

## ✨ Features

- 📚 **Multiple Input Formats** - EPUB, PDF, TXT support via Calibre
- 🎙️ **Dual TTS Engines** - Orpheus (preset voices) or Chatterbox (voice cloning)
- 🎭 **Emotion Enhancement** - AI-powered emotion tags for expressive narration
- 📖 **M4B Output** - Audiobooks with chapters and embedded cover art
- 🖥️ **Web UI & CLI** - User-friendly interface or command-line automation
- 🐳 **Docker-based** - Easy setup with automatic model downloads
- 🚀 **GPU Acceleration** - NVIDIA GPU support for faster generation

## 🚀 Quick Start

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Docker** | [Install Docker](https://docs.docker.com/get-docker/) (includes Docker Compose) |
| **Disk Space** | ~30GB free (models + generated audiobooks) |
| **RAM** | 16GB minimum (32GB recommended for CPU-only) |
| **GPU (Optional)** | NVIDIA GPU with 8GB+ VRAM for 10-50x faster generation |

### Step 1: Clone the Repository

```bash
git clone https://github.com/fhenwood/audiobook-creator.git
cd audiobook-creator
```

### Step 2: Configure Environment

```bash
# Copy the sample environment file
cp .env_sample .env

# (Optional) Edit settings - defaults work for most users
nano .env
```

Key settings in `.env`:
- `GPU_LAYERS=99` - Set to `0` for CPU-only, `99` for full GPU
- `DEFAULT_VOICE=zac` - Default Orpheus voice

### Step 3: Install NVIDIA Container Toolkit (GPU Users Only)

Skip this step if using CPU only.

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
</details>

<details>
<summary><b>Fedora/RHEL/CentOS</b></summary>

```bash
# Add NVIDIA package repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install the toolkit
sudo dnf install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
</details>

<details>
<summary><b>Arch Linux</b></summary>

```bash
# Install from AUR
yay -S nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
</details>

Verify installation:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Step 4: Start the Application

**With GPU (Recommended):**
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
```

**CPU Only:**
```bash
docker compose -f docker-compose.yaml -f docker-compose.cpu.yaml up -d
```

### Step 5: Wait for Model Downloads

First run downloads AI models (~15GB). Monitor progress:

```bash
# Watch download progress
docker compose logs -f model_downloader

# Watch main application logs
docker compose logs -f audiobook_creator
```

### Step 6: Access the Web UI

Open **http://localhost:7860** in your browser.

> ⏳ **Note:** Chatterbox (voice cloning) downloads an additional ~3GB model on first use.

### Verify GPU is Working

```bash
docker exec audiobook-creator-audiobook_creator-1 python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Stop the Application

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml down
```

## 📖 Usage

### Web Interface

1. **Upload Book** - Select EPUB, PDF, or TXT file
2. **Extract Text** - Converts book to plain text using Calibre
3. **Add Emotion Tags** (Optional) - Enhances text with expressive markers
4. **Select TTS Engine**:
   - **Orpheus**: Choose from 8 preset voices
   - **Chatterbox**: Upload a reference audio for voice cloning
5. **Generate Audiobook** - Choose MP3, WAV, or M4B format

### CLI Commands

Run commands inside the Docker container:

```bash
# Enter the container
docker compose exec audiobook_creator bash
```

#### Using the CLI Tool

```bash
# Extract text from an ebook
python cli.py extract /app/path/to/book.epub

# Add emotion tags (requires extracted text)
python cli.py emotion-tags

# Generate audiobook with Orpheus
python cli.py generate /app/path/to/book.epub --voice zac --format m4b

# Generate audiobook with emotion tags
python cli.py generate /app/path/to/book.epub --voice tara --emotion-tags

# Generate audiobook with Chatterbox (voice cloning)
python cli.py generate /app/path/to/book.epub --engine chatterbox --reference /app/audio_samples/voice.wav

# Generate a voice sample
python cli.py sample zac "Hello, this is a test."

# Get help
python cli.py --help
python cli.py generate --help
```

#### Direct Python Commands

You can also use Python directly for more control:

```bash
python -c "
from audiobook.core.text_extraction import extract_text_from_book_using_calibre
text = extract_text_from_book_using_calibre('/app/path/to/book.epub')
with open('converted_book.txt', 'w') as f:
    f.write(text)
print('Text extracted successfully!')
"
```

#### Add Emotion Tags

```bash
python -c "
import asyncio
from audiobook.core.emotion_tags import process_emotion_tags

async def main():
    async for progress in process_emotion_tags(False):
        print(progress)

asyncio.run(main())
"
```

#### Generate Audiobook with Orpheus

```bash
python -c "
import asyncio
from audiobook.tts.generator import process_audiobook_generation

async def main():
    async for progress in process_audiobook_generation(
        generation_mode='Single Voice',
        narrator_voice='zac',
        output_format='M4B (Chapters & Cover)',
        book_file_path='/app/path/to/book.epub',
        add_emotion_tags=True,
        tts_engine='Orpheus'
    ):
        print(progress)

asyncio.run(main())
"
```

#### Generate Audiobook with Chatterbox (Voice Cloning)

```bash
python -c "
import asyncio
from audiobook.tts.generator import process_audiobook_generation

async def main():
    async for progress in process_audiobook_generation(
        generation_mode='Single Voice',
        narrator_voice='cloned',
        output_format='MP3',
        book_file_path='/app/path/to/book.epub',
        add_emotion_tags=False,
        tts_engine='Chatterbox',
        reference_audio_path='/app/audio_samples/my_voice.wav'
    ):
        print(progress)

asyncio.run(main())
"
```

#### Preview Voice Samples

```bash
# Generate a voice sample with Orpheus
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://orpheus_tts:8880/v1', api_key='not-needed')
response = client.audio.speech.create(
    model='orpheus',
    voice='zac',
    input='Hello! This is a sample of my voice.',
    response_format='wav'
)
with open('sample.wav', 'wb') as f:
    f.write(response.content)
print('Sample saved to sample.wav')
"
```

## 🎙️ Available Voices (Orpheus)

| Voice | Gender | Description |
|-------|--------|-------------|
| Tara  | Female | Clear, professional |
| Leah  | Female | Warm, friendly |
| Jess  | Female | Energetic |
| Mia   | Female | Soft, gentle |
| Leo   | Male   | Deep, authoritative |
| Dan   | Male   | Conversational |
| Zac   | Male   | Natural, versatile |
| Zoe   | Female | Youthful |

## 🎭 Emotion Tags (Orpheus Only)

The LLM automatically adds expressive tags to enhance narration:

| Tag | Effect |
|-----|--------|
| `<laugh>` | Laughter |
| `<chuckle>` | Light laugh |
| `<sigh>` | Sighing |
| `<gasp>` | Surprise/shock |
| `<cough>` | Coughing |
| `<groan>` | Groaning |
| `<yawn>` | Yawning |
| `<sniffle>` | Sniffling |

## 🔧 Configuration

Edit `.env` to customize:

```bash
# ============ TTS Engine ============
# Orpheus voice (tara, leah, jess, leo, dan, mia, zac, zoe)
DEFAULT_VOICE=zac

# ============ GPU Settings ============
# GPU layers to offload (0=CPU only, 99=full GPU)
GPU_LAYERS=99

# Force CPU even with GPU available
FORCE_CPU=false

# ============ Model Selection ============
# Orpheus TTS model (quality vs speed)
ORPHEUS_MODEL_NAME=orpheus-3b-0.1-ft-Q8_0.gguf  # Best quality
# ORPHEUS_MODEL_NAME=orpheus-3b-0.1-ft-Q4_K_M.gguf  # Balanced
# ORPHEUS_MODEL_NAME=orpheus-3b-0.1-ft-Q2_K.gguf  # Fastest

# LLM for emotion tags (larger = better quality)
LLM_MODEL_FILE=gpt-oss-20b-Q4_K_M.gguf  # Best quality (~12GB)
# LLM_MODEL_FILE=Llama-3.2-3B-Instruct-Q4_K_M.gguf  # Smaller (~3GB)

# ============ Performance ============
# Parallel TTS requests (increase with more VRAM)
TTS_MAX_PARALLEL_REQUESTS_BATCH_SIZE=4

# LLM context size
LLM_CTX_SIZE=16384
```

## 🌍 Multilingual Support

Change `ORPHEUS_MODEL_NAME` in `.env` for other languages:

| Language | Model |
|----------|-------|
| English (default) | `orpheus-3b-0.1-ft-Q8_0.gguf` |
| French | `Orpheus-3b-French-FT-Q4_K_M.gguf` |
| German | `Orpheus-3b-German-FT-Q4_K_M.gguf` |
| Korean | `Orpheus-3b-Korean-FT-Q4_K_M.gguf` |
| Mandarin | `Orpheus-3b-Chinese-FT-Q4_K_M.gguf` |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│     docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml│
├───────────────────────────────────┬───────────────┬─────────────────┤
│         audiobook_creator         │ orpheus_llama │   llm_server    │
│    (Web UI + TTS Engines)         │ (TTS Tokens)  │   (GPT-OSS)     │
│        :7860 + :8880              │    :5006      │     :8000       │
│                                   │               │                 │
│  ┌─ Gradio Web UI (:7860)         │ - llama.cpp   │ - llama.cpp     │
│  ├─ Orpheus TTS API (:8880)       │ - GGUF model  │ - GGUF model    │
│  ├─ Chatterbox (voice cloning)    │ - GPU/CPU     │ - GPU/CPU       │
│  └─ SNAC audio codec              │               │                 │
└───────────────────────────────────┴───────────────┴─────────────────┘

Services:
• audiobook_creator - Main app with Gradio UI, Orpheus TTS API, and Chatterbox
• orpheus_llama     - llama.cpp server for Orpheus token generation  
• llm_server        - llama.cpp server for emotion tag generation (GPT-OSS-20B)
```

## 💻 Hardware Requirements

| Component | Minimum (CPU) | Recommended (GPU) |
|-----------|---------------|-------------------|
| **RAM** | 32GB | 16GB |
| **VRAM** | - | 8GB+ NVIDIA |
| **Storage** | 30GB | 50GB+ |
| **CPU** | 8 cores | 4+ cores |

### Generation Speed (Approximate)

| Hardware | Speed |
|----------|-------|
| CPU only | ~0.1x realtime |
| RTX 3060 (12GB) | ~2x realtime |
| RTX 4090 (24GB) | ~5x realtime |

## 🐛 Troubleshooting

### View Logs

```bash
# All services
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml logs -f

# Specific service
docker compose logs -f audiobook_creator   # Web UI + Orpheus TTS API + Chatterbox
docker compose logs -f orpheus_llama        # Orpheus token generation (llama.cpp)
docker compose logs -f llm_server           # Emotion tags LLM (GPT-OSS)
```

### Reset Models

```bash
# Stop all services
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml down

# Remove cached models and rebuild from scratch
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml down -v
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build -d
```

### Common Issues

**"Chatterbox model downloading..."** - First use downloads ~3GB model. Wait for completion.

**Out of VRAM** - Reduce `GPU_LAYERS` in `.env` or use CPU mode.

**Slow generation / Running on CPU** - Make sure you started with both compose files:
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
```
Verify GPU is available inside the container:
```bash
docker exec audiobook-creator-audiobook_creator-1 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Text extraction fails** - Ensure book file isn't DRM-protected.

**"GPU: No" in logs** - Container started without GPU override. Restart with `-f docker-compose.gpu.yaml`.

## 📁 Project Structure

```
audiobook-creator/
├── app.py                      # Main Gradio web interface
├── cli.py                      # Command-line interface
├── audiobook/                  # Main package
│   ├── __init__.py            # Package exports
│   ├── core/                  # Core processing modules
│   │   ├── text_extraction.py # Text extraction from ebooks
│   │   ├── emotion_tags.py    # Emotion tag enhancement
│   │   └── character_identification.py  # Character identification
│   ├── tts/                   # TTS generation modules
│   │   ├── generator.py       # Audiobook generation logic
│   │   ├── voice_mapping.py   # Voice configuration
│   │   └── audio_utils.py     # M4B/audio processing
│   └── utils/                 # Utility modules
│       ├── file_utils.py      # File operations
│       ├── llm_utils.py       # LLM interaction
│       ├── text_preprocessing.py  # Text cleanup
│       ├── shell_commands.py  # Shell command execution
│       └── emotion_processing.py  # Emotion tag processing
├── orpheus_tts/               # Orpheus TTS service
├── generated_audiobooks/      # Output directory
├── audio_samples/             # Voice samples & references
├── sample_book_and_audio/     # Example book and audiobooks
├── static_files/              # Static assets (voice map, etc.)
├── docker-compose.yaml        # Docker orchestration
├── Dockerfile                 # Container build instructions
└── requirements.txt           # Python dependencies
```

## 📜 License

GNU General Public License v3.0

## 🙏 Credits

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) - Canopy Labs
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - Resemble AI
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Cross-platform LLM inference
- [GPT-OSS-20B](https://huggingface.co/cognitivecomputations/gpt-oss-20b) - Open source LLM
- [Calibre](https://calibre-ebook.com/) - Ebook conversion

---

Made with ❤️ for audiobook enthusiasts
