# JAX - Spark Pack Voice Assistant

JAX is an AI voice assistant for controlling OpenArm robots through natural language. It uses NVIDIA Riva for speech recognition and synthesis, and Llama 3.1 via NVIDIA NIM for natural language understanding.

## Features

- **Voice Commands**: Control robot teleoperation with natural speech
- **Chat Mode**: Conversational AI with Jarvis-like personality
- **Web Interface**: Control from any device (iPhone, tablet)
- **Local Processing**: All AI runs locally on NVIDIA GPUs

## Prerequisites

- NVIDIA GPU (RTX 4090 or better recommended)
- Docker with NVIDIA Container Toolkit
- NGC API Key (free from [NGC](https://ngc.nvidia.com/))
- Python 3.10+

## Installation

### 1. Install NVIDIA Riva (ASR/TTS)

```bash
# Download Riva quickstart (2.19 has Magpie TTS - more natural voices)
ngc registry resource download-version nvidia/riva/riva_quickstart:2.19.0

# Navigate to Riva directory
cd ~/riva/riva_quickstart_v2.19.0

# Configure for Magpie TTS (already configured if using our setup):
# tts_model=("magpie")
# tts_language_code=("multi")

# Initialize (downloads Magpie models - takes a while)
bash riva_init.sh

# Start Riva server
bash riva_start.sh
```

**Note:** Magpie TTS provides more natural, expressive voices with emotional tones (Calm, Happy, Neutral, etc.).

Verify Riva is running:
```bash
curl localhost:50051  # Should connect (gRPC endpoint)
```

### 2. Install NVIDIA NIM (LLM)

```bash
cd /path/to/openarm_isaac_lab_trainer/scripts/voice_assistant

# Start NIM container (Llama 3.1 8B)
./nim_start.sh
```

This will:
- Pull the NIM container from NGC
- Download Llama 3.1 8B weights (~8GB)
- Start the API on port 8000

Verify NIM is running:
```bash
curl http://localhost:8000/v1/models
```

### 3. Install Python Dependencies

```bash
cd /path/to/openarm_isaac_lab_trainer

# Using uv (recommended)
uv add nvidia-riva-client aiohttp requests

# Or using pip
pip install nvidia-riva-client aiohttp requests
```

### 4. System Audio Setup

```bash
# Install PortAudio (required for microphone)
sudo apt install portaudio19-dev

# List available microphones
pactl list sources short

# Set default microphone (replace with your device)
pactl set-default-source alsa_input.usb-YOUR_MIC_DEVICE
```

## Running JAX

### Option 1: Standalone Voice Assistant

Direct microphone interaction - speak and JAX responds verbally.

```bash
cd scripts
./voice_assistant.sh
```

**Commands:**
- "Start recording" / "Stop recording"
- "Drop a lemon" / "Spawn an orange"
- "Reset objects"
- "The task is [description]"
- Or just chat: "Hello JAX, how are you?"

### Option 2: Web Interface

Access JAX from any device on your network (iPhone, tablet, etc.)

```bash
cd scripts
./voice_web.sh
```

Then open `http://YOUR_IP:8080` in a browser.

**Features:**
- Text input for typing commands
- Voice recording (tap microphone button)
- Quick command buttons
- Real-time responses

### Option 3: Conversation Test

Simple back-and-forth conversation test.

```bash
cd scripts
./voice_test.sh
```

## Configuration

### Command Line Options

**voice_assistant.sh / voice_server.py:**
```bash
--riva-server localhost:50051    # Riva server address
--nim-url http://localhost:8000  # NIM API endpoint
--zmq-port 5556                  # ZMQ port for teleop
--no-tts                         # Disable voice output
--no-llm                         # Use keyword parsing only
```

**voice_web.sh / web_server.py:**
```bash
--port 8080                      # Web server port
--riva-server localhost:50051    # Riva server address
--nim-url http://localhost:8000  # NIM API endpoint
```

### Remote Services

Run Riva/NIM on a different machine (e.g., NVIDIA Thor):

```bash
# Point to remote servers
./voice_web.sh --riva-server thor:50051 --nim-url http://thor:8000
```

### Using 70B Model (Better Quality)

For systems with 40GB+ VRAM (e.g., NVIDIA Thor):

```bash
# On the Thor/server:
./nim_start_70b.sh

# On your workstation:
./voice_web.sh --nim-url http://thor:8000
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Microphone    │────▶│   Riva ASR      │
└─────────────────┘     │   (Speech→Text) │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   NIM (Llama)   │
                        │   (NLU/Chat)    │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐
     │   Riva TTS      │ │   ZMQ Publish   │ │  Web UI     │
     │   (Text→Speech) │ │   (to Teleop)   │ │  Response   │
     └─────────────────┘ └─────────────────┘ └─────────────┘
```

## GPU Memory Usage

| Component | VRAM |
|-----------|------|
| Riva ASR/TTS (Magpie) | ~4-6 GB |
| NIM 8B (FP8) | ~10 GB |
| NIM 70B | ~40 GB |

For single RTX 4090 (24GB):
- Riva + NIM 8B fits comfortably
- Use `--no-llm` if running other workloads

## Troubleshooting

### "PortAudio library not found"
```bash
sudo apt install portaudio19-dev
```

### "Riva ASR not available"
```bash
# Check Riva is running
docker ps | grep riva
# Restart if needed
cd ~/riva/riva_quickstart_v2.19.0
bash riva_stop.sh && bash riva_start.sh
```

### "NIM not available"
```bash
# Check NIM is running
curl http://localhost:8000/v1/models
# Restart if neededEE
./nim_start.sh
```

### Microphone not working
```bash
# List sources
pactl list sources short

# Test recording
parecord --device=YOUR_DEVICE -d 3 test.wav
aplay test.wav

# Set default
pactl set-default-source YOUR_DEVICE
```

### NIM out of memory
```bash
# Reduce context length
docker run ... -e NIM_MAX_MODEL_LEN=4096 ...
```

## Files

| File | Description |
|------|-------------|
| `voice_server.py` | Main voice assistant (microphone) |
| `web_server.py` | Web interface server |
| `voice_test.py` | Conversation test script |
| `voice_assistant.sh` | Launcher for voice_server.py |
| `voice_web.sh` | Launcher for web_server.py |
| `voice_test.sh` | Launcher for voice_test.py |
| `nim_start.sh` | Start NIM 8B container |
| `nim_start_70b.sh` | Start NIM 70B container |
| `riva_setup.sh` | Riva installation helper |

## License

Part of the Spark Pack OpenArm project.
