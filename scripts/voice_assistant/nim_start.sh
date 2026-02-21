#!/bin/bash
# Launch NVIDIA NIM for Llama 3.1 8B Instruct
# This provides TensorRT-optimized inference for JAX

set -e

echo "=================================================="
echo "NVIDIA NIM - Llama 3.1 8B Instruct"
echo "=================================================="

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    # Try to get from ngc config
    if command -v ngc &> /dev/null; then
        NGC_API_KEY=$(grep -A2 '\[CURRENT\]' ~/.ngc/config 2>/dev/null | grep apikey | cut -d'=' -f2 | tr -d ' ' || true)
    fi
    
    if [ -z "$NGC_API_KEY" ]; then
        echo "ERROR: NGC_API_KEY not set"
        echo ""
        echo "Set it with:"
        echo "  export NGC_API_KEY=<your-api-key>"
        echo ""
        echo "Or run 'ngc config set' to configure NGC CLI"
        exit 1
    fi
fi

# Login to NGC container registry
echo ""
echo "[1/3] Logging into NGC container registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Create cache directory for model weights
NIM_CACHE="${HOME}/.cache/nim"
mkdir -p "$NIM_CACHE"

# Check if container is already running
CONTAINER_NAME="nim-llama"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "NIM container is already running!"
    echo "API endpoint: http://localhost:8000/v1/chat/completions"
    echo ""
    echo "To stop it: docker stop ${CONTAINER_NAME}"
    exit 0
fi

# Remove old stopped container if exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "[2/3] Starting NIM container (this may download ~15GB on first run)..."
echo ""

# Run NIM container
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -v "$NIM_CACHE:/opt/nim/.cache" \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

echo ""
echo "[3/3] Waiting for NIM to initialize..."
echo ""

# Wait for server to be ready
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo ""
        echo "=================================================="
        echo "NIM READY!"
        echo "=================================================="
        echo ""
        echo "API endpoint: http://localhost:8000/v1/chat/completions"
        echo "Model: meta/llama-3.1-8b-instruct"
        echo ""
        echo "Test with:"
        echo '  curl -X POST http://localhost:8000/v1/chat/completions \'
        echo '    -H "Content-Type: application/json" \'
        echo '    -d '"'"'{"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'"'"
        echo ""
        echo "To view logs: docker logs -f ${CONTAINER_NAME}"
        echo "To stop:      docker stop ${CONTAINER_NAME}"
        exit 0
    fi
    
    # Show progress
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  Waiting... ($WAITED seconds)"
        # Show last few log lines
        docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | grep -v "^$" | head -1 || true
    fi
    
    sleep 2
    WAITED=$((WAITED + 2))
done

echo ""
echo "WARNING: NIM is still initializing after ${MAX_WAIT}s"
echo "This is normal on first run (downloading model weights)"
echo ""
echo "Check progress with: docker logs -f ${CONTAINER_NAME}"
echo "Container will be ready when you see 'Uvicorn running on'"
