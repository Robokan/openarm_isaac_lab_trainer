#!/usr/bin/env bash
# Voice Assistant Test - Talk and Listen
#
# Simple conversational test for the voice assistant.
# No robot control - just talk and hear responses.
#
# Prerequisites:
#   - NVIDIA Riva running (see scripts/voice_assistant/riva_setup.sh)
#   - NVIDIA NIM running for LLM (see scripts/voice_assistant/nim_start.sh)
#   - Microphone connected
#
# Usage:
#   ./scripts/voice_test.sh
#   ./scripts/voice_test.sh --no-llm  # Simple responses without NIM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENPI_DIR="/home/bizon/sparkpack/openpi"

echo "========================================"
echo "JAX VOICE ASSISTANT TEST"
echo "========================================"
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
    echo "If Riva is not installed yet, run:"
    echo "  ${SCRIPT_DIR}/voice_assistant/riva_setup.sh"
    echo ""
fi

# Check if NIM is running (unless --no-llm)
if [[ "$*" != *"--no-llm"* ]]; then
    if docker ps 2>/dev/null | grep -q nim-llama; then
        echo "[OK] NIM server is running (LLM)"
    else
        echo "[WARN] NIM server (LLM) may not be running"
        echo ""
        echo "To start NIM:"
        echo "  ${SCRIPT_DIR}/voice_assistant/nim_start.sh"
        echo ""
        echo "Or run without LLM using simple responses:"
        echo "  $0 --no-llm"
        echo ""
    fi
fi

# Check for microphone
echo ""
echo "Checking audio devices..."
if command -v arecord &> /dev/null; then
    arecord -l 2>/dev/null || echo "  No ALSA devices found"
fi

# Use OpenPi's uv environment
cd "$OPENPI_DIR"

# Install voice assistant dependencies via uv
echo ""
echo "Checking dependencies..."
uv add --quiet nvidia-riva-client sounddevice requests 2>/dev/null || true

echo ""
echo "Starting voice assistant test..."
echo ""

# Run test using uv
uv run python "${SCRIPT_DIR}/voice_assistant/voice_test.py" "$@"
