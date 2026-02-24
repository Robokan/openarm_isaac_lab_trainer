#!/bin/bash
# Start all voice assistant services (Riva + NIM + Voice Server)
#
# Usage:
#   ./start_all.sh          # Start everything
#   ./start_all.sh --web    # Start with web interface instead of microphone
#
# Prerequisites:
#   - Riva must be installed: ~/riva/riva_quickstart_v2.18.0
#   - NGC_API_KEY environment variable set
#
# This will start:
#   1. Riva ASR/TTS (if not already running)
#   2. NIM LLM (if not already running)  
#   3. Voice assistant (microphone or web interface)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "  JAX Voice Assistant - Full Startup"
echo "========================================"
echo ""

# Parse arguments
USE_WEB=false
for arg in "$@"; do
    case $arg in
        --web)
            USE_WEB=true
            shift
            ;;
    esac
done

# Check for Riva
RIVA_DIR="$HOME/riva/riva_quickstart_v2.18.0"
if [ ! -d "$RIVA_DIR" ]; then
    RIVA_DIR="$HOME/riva/riva_quickstart_v2.19.0"
fi

if [ ! -d "$RIVA_DIR" ]; then
    echo -e "${RED}[ERROR] Riva not found. Install from:${NC}"
    echo "  ngc registry resource download-version nvidia/riva/riva_quickstart:2.18.0"
    exit 1
fi

# ============================================
# 1. Start Riva (if not running)
# ============================================
echo -e "${YELLOW}[1/3] Checking Riva...${NC}"
if docker ps | grep -q "riva-speech"; then
    echo -e "${GREEN}  Riva already running${NC}"
else
    echo "  Starting Riva (this may take a minute)..."
    cd "$RIVA_DIR"
    bash riva_start.sh &
    
    # Wait for Riva to be ready
    echo "  Waiting for Riva to initialize..."
    for i in {1..60}; do
        if timeout 1 bash -c "echo > /dev/tcp/localhost/50051" 2>/dev/null; then
            echo -e "${GREEN}  Riva ready!${NC}"
            break
        fi
        sleep 2
        echo "    Waiting... ($((i*2))s)"
    done
fi

# ============================================
# 2. Start NIM (if not running)
# ============================================
echo ""
echo -e "${YELLOW}[2/3] Checking NIM...${NC}"
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}  NIM already running${NC}"
else
    echo "  Starting NIM (first run downloads ~8GB model)..."
    
    # Check for NGC API key
    if [ -z "$NGC_API_KEY" ]; then
        if [ -f "$HOME/.ngc/config" ]; then
            export NGC_API_KEY=$(grep apikey "$HOME/.ngc/config" | cut -d'=' -f2 | tr -d ' ')
        fi
    fi
    
    if [ -z "$NGC_API_KEY" ]; then
        echo -e "${RED}  [ERROR] NGC_API_KEY not set${NC}"
        echo "  Get your key from: https://ngc.nvidia.com/"
        echo "  Then: export NGC_API_KEY=your_key"
        exit 1
    fi
    
    # Start NIM in background
    docker run -d --rm \
        --name nim-llm \
        --gpus '"device=1"' \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$HOME/.cache/nim:/home/user/.cache" \
        -e NGC_API_KEY="$NGC_API_KEY" \
        -e NIM_MAX_MODEL_LEN=8192 \
        -e NIM_GPU_MEMORY_UTILIZATION=0.9 \
        -p 8000:8000 \
        nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
    
    # Wait for NIM to be ready
    echo "  Waiting for NIM to initialize (can take 2-5 minutes on first run)..."
    for i in {1..120}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo -e "${GREEN}  NIM ready!${NC}"
            break
        fi
        sleep 5
        echo "    Waiting... ($((i*5))s)"
    done
    
    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo -e "${RED}  [ERROR] NIM failed to start. Check: docker logs nim-llm${NC}"
        exit 1
    fi
fi

# ============================================
# 3. Start Voice Assistant
# ============================================
echo ""
echo -e "${YELLOW}[3/3] Starting Voice Assistant...${NC}"
cd "$PROJECT_DIR/scripts"

if [ "$USE_WEB" = true ]; then
    echo "  Mode: Web Interface (http://localhost:8080)"
    ./voice_web.sh
else
    echo "  Mode: Microphone"
    ./voice_assistant.sh
fi
