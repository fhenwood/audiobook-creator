# Audiobook Creator

Convert books (EPUB, PDF, TXT) into fully voiced audiobooks using Orpheus TTS and GPT-OSS-20B.

**Works on macOS, Linux, and Windows** with automatic GPU detection.

## Features

- 🎙️ **Orpheus TTS** - High-quality, expressive text-to-speech with emotion tags
- 🤖 **GPT-OSS-20B** - Character identification and emotion enhancement
- 🚀 **GPU Auto-Detection** - Uses NVIDIA GPU if available, falls back to CPU
- 🐳 **Cross-Platform** - Mac (Apple Silicon/Intel), Linux, Windows
- 🎧 **Voice Preview** - Sample all 8 voices before creating audiobooks
- 📖 **M4B Support** - Audiobooks with chapters and cover art

## Quick Start

```bash
git clone https://github.com/prakharsr/audiobook-creator.git
cd audiobook-creator
cp .env_sample .env
docker compose up --build
```

Open **http://localhost:7860**

> First run downloads ~15GB of models (Orpheus ~3GB + GPT-OSS-20B ~12GB).

### Force CPU Mode

If you have a GPU but want CPU-only:
```bash
# Option 1: Set in .env
FORCE_CPU=true

# Option 2: Use override file
docker compose -f docker-compose.yaml -f docker-compose.cpu.yaml up --build
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    docker compose up                         │
├───────────────┬───────────────┬───────────────┬─────────────┤
│ audiobook_    │  orpheus_tts  │ orpheus_llama │ llm_server  │
│ creator       │  (TTS API)    │ (TTS Tokens)  │ (GPT-OSS)   │
│ Port: 7860    │  Port: 8880   │ Port: 5006    │ Port: 8000  │
└───────────────┴───────────────┴───────────────┴─────────────┘
        All services use llama.cpp (cross-platform)
```

## Available Voices

| Voice | Gender | | Voice | Gender |
|-------|--------|-|-------|--------|
| Tara | Female | | Leo | Male |
| Leah | Female | | Dan | Male |
| Jess | Female | | Zac | Male |
| Mia | Female | | Zoe | Female |

## Multilingual Models

Change `ORPHEUS_MODEL_NAME` in `.env`:
- `Orpheus-3b-French-FT-Q4_K_M.gguf` - French
- `Orpheus-3b-German-FT-Q4_K_M.gguf` - German  
- `Orpheus-3b-Korean-FT-Q4_K_M.gguf` - Korean
- `Orpheus-3b-Chinese-FT-Q4_K_M.gguf` - Mandarin

## Emotion Tags

The LLM automatically adds expressive tags:
`<laugh>` `<chuckle>` `<sigh>` `<gasp>` `<cough>` `<groan>` `<yawn>`

## Hardware Requirements

| | Minimum (CPU) | Recommended (GPU) |
|---|---------------|-------------------|
| **RAM** | 32GB | 16GB |
| **VRAM** | - | 8GB+ |
| **Storage** | 30GB | 50GB+ |
| **CPU** | 8 cores | 4+ cores |

## Configuration

Key settings in `.env`:

```bash
# Force CPU mode even with GPU available
FORCE_CPU=false

# GPU layers (99=all, reduce if limited VRAM)
LLM_GPU_LAYERS=99
ORPHEUS_GPU_LAYERS=99

# Use smaller LLM for limited hardware
LLM_MODEL_FILE=Llama-3.2-3B-Instruct-Q4_K_M.gguf
LLM_HF_REPO=bartowski/Llama-3.2-3B-Instruct-GGUF

# Adjust threads for CPU mode
LLM_THREADS=8
ORPHEUS_THREADS=4
```

## Troubleshooting

```bash
# View logs
docker compose logs -f llm_server
docker compose logs -f orpheus_tts

# Reset and re-download models
docker compose down -v
```

## Credits

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) - Canopy Labs
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Cross-platform LLM inference
- [GPT-OSS-20B](https://huggingface.co/cognitivecomputations/gpt-oss-20b) - Open source LLM

## License

GNU General Public License v3.0
