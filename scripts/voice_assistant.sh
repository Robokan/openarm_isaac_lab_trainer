#!/usr/bin/env bash
# Voice Assistant Launcher for OpenArm Teleoperation
#
# Starts the voice assistant server.
# The server listens for voice commands and sends them to the teleoperation client via ZMQ.
#
# Prerequisites:
#   - NVIDIA Riva running (see scripts/voice_assistant/riva_setup.sh)
#   - NVIDIA NIM running for LLM (see scripts/voice_assistant/nim_start.sh)
#   - Microphone connected
#
# Usage:
#   ./scripts/voice_assistant.sh
#   ./scripts/voice_assistant.sh --no-llm  # Keyword matching only (no NIM)
#
# To start services first (in separate terminals):
#   Terminal 1: cd ~/riva/riva_quickstart_v2.18.0 && bash riva_start.sh
#   Terminal 2: ./scripts/voice_assistant/nim_start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENPI_DIR="/home/bizon/sparkpack/openpi"

echo "========================================"
echo "JAX - SPARK PACK VOICE ASSISTANT"
echo "========================================"
echo ""
echo "ZMQ Port: 5556"
echo ""

# Check if Riva is running
if docker ps 2>/dev/null | grep -q riva; then
    echo "[OK] Riva server is running (ASR/TTS)"
else
    echo "[WARN] Riva server may not be running"
    echo ""
    echo "To start Riva:"
    echo "  cd ~/riva/riva_quickstart_v2.18.0"
    echo "  bash riva_start.sh"
    echo ""
    echo "If Riva is not installed, run:"
    echo "  ${SCRIPT_DIR}/voice_assistant/riva_setup.sh"
    echo ""
fi

# Check if NIM is running (unless --no-llm)
if [[ "$*" != *"--no-llm"* ]]; then
    if docker ps 2>/dev/null | grep -qE "nim-llama|trtllm-llama"; then
        echo "[OK] LLM server is running"
    else
        echo "[WARN] LLM server may not be running"
        echo ""
        echo "To start LLM (DGX Spark 70B):"
        echo "  ${SCRIPT_DIR}/voice_assistant/nim_start_70b_spark.sh"
        echo ""
        echo "To start LLM (8B):"
        echo "  ${SCRIPT_DIR}/voice_assistant/nim_start.sh"
        echo ""
        echo "Or run without LLM using keyword matching:"
        echo "  $0 --no-llm"
        echo ""
    fi
fi

# Check for microphone
if ! command -v arecord &> /dev/null; then
    echo "[WARN] 'arecord' not found - cannot check microphone"
else
    MIC_COUNT=$(arecord -l 2>/dev/null | grep -c "^card" || echo "0")
    if [ "$MIC_COUNT" -eq "0" ]; then
        echo "[WARN] No microphone detected!"
        echo "  Connect a USB microphone or headset"
        echo ""
    else
        echo "[OK] Found $MIC_COUNT audio input device(s)"
    fi
fi

# Use OpenPi's uv environment
cd "$OPENPI_DIR"

# Install voice assistant dependencies via uv
echo ""
echo "Checking dependencies..."
uv add --quiet nvidia-riva-client sounddevice pyzmq requests 2>/dev/null || true

echo ""
echo "Starting voice server..."
echo "Press Ctrl+C to quit"
echo ""

# Run voice server using uv
uv run python "${SCRIPT_DIR}/voice_assistant/voice_server.py" "$@"
