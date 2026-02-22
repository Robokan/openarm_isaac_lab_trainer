#!/bin/bash
# NVIDIA Riva ASR + TTS Setup for DGX Spark
#
# Uses Riva NIM containers (Parakeet CTC for ASR, FastPitch for TTS)
# optimized for the GB10 Grace Blackwell Superchip.
#
# Prerequisites:
#   - DGX Spark with NVIDIA drivers
#   - Docker with NVIDIA Container Toolkit
#   - NGC API Key: export NGC_API_KEY=<your-key>
#
# Usage:
#   ./riva_setup_spark.sh          # Start both ASR and TTS
#   ./riva_setup_spark.sh --asr    # Start ASR only
#   ./riva_setup_spark.sh --tts    # Start TTS only
#   ./riva_setup_spark.sh --stop   # Stop all Riva containers

set -e

echo "=================================================="
echo "NVIDIA Riva ASR + TTS - DGX Spark"
echo "=================================================="

ASR_CONTAINER="riva-asr-spark"
TTS_CONTAINER="riva-tts-spark"
ASR_IMAGE="nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:latest"
TTS_IMAGE="nvcr.io/nim/nvidia/fastpitch-hifigan-tts-en-us:latest"
GRPC_PORT=50051
ASR_HTTP_PORT=9000
TTS_HTTP_PORT=9001

START_ASR=true
START_TTS=true

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asr)
            START_TTS=false
            shift
            ;;
        --tts)
            START_ASR=false
            shift
            ;;
        --stop)
            echo "Stopping Riva containers..."
            docker stop "$ASR_CONTAINER" "$TTS_CONTAINER" 2>/dev/null || true
            docker rm "$ASR_CONTAINER" "$TTS_CONTAINER" 2>/dev/null || true
            echo "Done."
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    if command -v ngc &> /dev/null; then
        NGC_API_KEY=$(grep -A2 '\[CURRENT\]' ~/.ngc/config 2>/dev/null | grep apikey | cut -d'=' -f2 | tr -d ' ' || true)
    fi
    
    if [ -z "$NGC_API_KEY" ]; then
        echo ""
        echo "ERROR: NGC_API_KEY not set"
        echo ""
        echo "Get a key from: https://ngc.nvidia.com/setup/api-key"
        echo "Then set it with:"
        echo "  export NGC_API_KEY=<your-api-key>"
        exit 1
    fi
fi

# Login to NGC
echo ""
echo "Logging into NGC container registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Create cache directory
NIM_CACHE="${HOME}/.cache/nim"
mkdir -p "$NIM_CACHE"

# ===== ASR (Speech-to-Text) =====
if [ "$START_ASR" = true ]; then
    echo ""
    echo "=========================================="
    echo "Starting Riva ASR (Parakeet CTC 1.1B)"
    echo "=========================================="
    
    if docker ps --format '{{.Names}}' | grep -q "^${ASR_CONTAINER}$"; then
        echo "ASR container already running!"
    else
        docker rm -f "$ASR_CONTAINER" 2>/dev/null || true
        
        docker run -d \
            --name "$ASR_CONTAINER" \
            --runtime=nvidia \
            --gpus '"device=0"' \
            --shm-size=8GB \
            -e NGC_API_KEY="$NGC_API_KEY" \
            -e NIM_HTTP_API_PORT="$ASR_HTTP_PORT" \
            -e NIM_GRPC_API_PORT="$GRPC_PORT" \
            -e "NIM_TAGS_SELECTOR=name=parakeet-1-1b-ctc-en-us,mode=all" \
            -p "$ASR_HTTP_PORT:$ASR_HTTP_PORT" \
            -p "$GRPC_PORT:$GRPC_PORT" \
            -v "$NIM_CACHE:/opt/nim/.cache" \
            "$ASR_IMAGE"
        
        echo "ASR container started. Waiting for initialization..."
        
        MAX_WAIT=300
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:${ASR_HTTP_PORT}/v1/health/ready" 2>/dev/null | grep -q "ready"; then
                echo ""
                echo "[OK] Riva ASR is ready!"
                echo "  gRPC: localhost:${GRPC_PORT}"
                echo "  HTTP: localhost:${ASR_HTTP_PORT}"
                break
            fi
            if [ $((WAITED % 15)) -eq 0 ]; then
                echo "  Waiting... ($WAITED seconds)"
            fi
            sleep 5
            WAITED=$((WAITED + 5))
        done
        
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[WARN] ASR still initializing. Check: docker logs -f ${ASR_CONTAINER}"
        fi
    fi
fi

# ===== TTS (Text-to-Speech) =====
if [ "$START_TTS" = true ]; then
    echo ""
    echo "=========================================="
    echo "Starting Riva TTS (FastPitch HiFiGAN)"
    echo "=========================================="
    
    if docker ps --format '{{.Names}}' | grep -q "^${TTS_CONTAINER}$"; then
        echo "TTS container already running!"
    else
        docker rm -f "$TTS_CONTAINER" 2>/dev/null || true
        
        # TTS uses a different gRPC port to avoid conflict with ASR
        TTS_GRPC_PORT=50052
        
        docker run -d \
            --name "$TTS_CONTAINER" \
            --runtime=nvidia \
            --gpus '"device=0"' \
            --shm-size=8GB \
            -e NGC_API_KEY="$NGC_API_KEY" \
            -e NIM_HTTP_API_PORT="$TTS_HTTP_PORT" \
            -e NIM_GRPC_API_PORT="$TTS_GRPC_PORT" \
            -p "$TTS_HTTP_PORT:$TTS_HTTP_PORT" \
            -p "$TTS_GRPC_PORT:$TTS_GRPC_PORT" \
            -v "$NIM_CACHE:/opt/nim/.cache" \
            "$TTS_IMAGE"
        
        echo "TTS container started. Waiting for initialization..."
        
        MAX_WAIT=300
        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "http://localhost:${TTS_HTTP_PORT}/v1/health/ready" 2>/dev/null | grep -q "ready"; then
                echo ""
                echo "[OK] Riva TTS is ready!"
                echo "  gRPC: localhost:${TTS_GRPC_PORT}"
                echo "  HTTP: localhost:${TTS_HTTP_PORT}"
                break
            fi
            if [ $((WAITED % 15)) -eq 0 ]; then
                echo "  Waiting... ($WAITED seconds)"
            fi
            sleep 5
            WAITED=$((WAITED + 5))
        done
        
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[WARN] TTS still initializing. Check: docker logs -f ${TTS_CONTAINER}"
        fi
    fi
fi

echo ""
echo "=================================================="
echo "Riva Setup Complete"
echo "=================================================="
echo ""
echo "Services:"
echo "  ASR (speech-to-text): gRPC localhost:${GRPC_PORT}"
echo "  TTS (text-to-speech): gRPC localhost:50052"
echo ""
echo "Start the voice assistant with:"
echo "  ./scripts/voice_web.sh --riva-server localhost:${GRPC_PORT} --tts-server localhost:50052"
echo ""
echo "Or with the 70B LLM:"
echo "  ./scripts/voice_assistant/nim_start_70b_spark.sh  # In another terminal"
echo "  ./scripts/voice_web.sh --riva-server localhost:${GRPC_PORT} --tts-server localhost:50052 --nim-url http://localhost:8000"
echo ""
echo "To stop Riva:"
echo "  ./scripts/voice_assistant/riva_setup_spark.sh --stop"
