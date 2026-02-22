#!/usr/bin/env bash
# JAX Web Interface
#
# Access JAX from your iPhone or any browser on your network.
#
# Prerequisites:
#   - NVIDIA Riva running (for ASR/TTS)
#   - NVIDIA NIM running (for LLM) - optional, falls back to keywords
#   - Teleop running (optional, for robot commands)
#
# Usage:
#   ./scripts/voice_web.sh
#   ./scripts/voice_web.sh --port 8080
#
# For remote services (e.g., Thor):
#   ./scripts/voice_web.sh --riva-server thor:50051 --nim-url http://thor:8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENPI_DIR="/home/bizon/sparkpack/openpi"

echo "========================================"
echo "JAX WEB INTERFACE"
echo "========================================"
echo ""

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "Access from your iPhone:"
echo "  http://${LOCAL_IP}:8080"
echo ""

# Check if Riva is running locally (skip if using remote)
if [[ "$*" != *"--riva-server"* ]]; then
    if docker ps 2>/dev/null | grep -q riva; then
        echo "[OK] Riva server is running locally"
    else
        echo "[WARN] Riva server may not be running locally"
        echo "  Start with: cd ~/riva/riva_quickstart_v2.18.0 && bash riva_start.sh"
        echo ""
    fi
fi

# Check if NIM is running locally (skip if using remote)
if [[ "$*" != *"--nim-url"* ]]; then
    if docker ps 2>/dev/null | grep -qE "nim-llama|trtllm-llama"; then
        echo "[OK] LLM server is running locally"
    else
        echo "[INFO] LLM not running locally - using keyword matching"
        echo "  For LLM (DGX Spark): ./scripts/voice_assistant/nim_start_70b_spark.sh"
        echo "  For LLM (8B):        ./scripts/voice_assistant/nim_start.sh"
        echo ""
    fi
fi

# Use OpenPi's uv environment
cd "$OPENPI_DIR"

# Install dependencies
echo "Checking dependencies..."
uv add --quiet aiohttp nvidia-riva-client pyzmq requests 2>/dev/null || true

echo ""
echo "Starting web server..."
echo ""

# Run web server
uv run python "${SCRIPT_DIR}/voice_assistant/web_server.py" "$@"
