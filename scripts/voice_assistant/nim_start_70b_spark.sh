#!/bin/bash
# Launch Llama 3.3 70B Instruct on DGX Spark via TensorRT-LLM
#
# Uses NVFP4 quantization (native to Blackwell) for optimal performance
# on the GB10 Grace Blackwell Superchip with 128GB unified memory.
#
# Provides an OpenAI-compatible API on port 8000, drop-in replacement
# for the NIM container used by the voice assistant.
#
# Prerequisites:
#   - DGX Spark with NVIDIA drivers
#   - Docker with NVIDIA Container Toolkit
#   - Hugging Face token: export HF_TOKEN=<your-token>
#
# Usage:
#   ./nim_start_70b_spark.sh
#
# Then point the voice assistant at it:
#   ./voice_web.sh --nim-url http://localhost:8000

set -e

echo "=================================================="
echo "Llama 3.3 70B Instruct (NVFP4) - DGX Spark"
echo "=================================================="
echo ""
echo "Container: TensorRT-LLM (optimized for Blackwell)"
echo "Model:     nvidia/Llama-3.3-70B-Instruct-FP4"
echo "Quant:     NVFP4 (4-bit, native Blackwell Tensor Cores)"
echo ""

DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"
MODEL_HANDLE="nvidia/Llama-3.3-70B-Instruct-FP4"
CONTAINER_NAME="trtllm-llama-70b"
PORT=8000

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo ""
    echo "Get a token from https://huggingface.co/settings/tokens"
    echo "Then set it with:"
    echo "  export HF_TOKEN=<your-token>"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  GPU: ${GPU_NAME}"
    echo "  Memory: ${GPU_MEM} MB"
else
    echo "  WARNING: nvidia-smi not found"
fi

# Check if already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Container '${CONTAINER_NAME}' is already running!"
    echo "API endpoint: http://localhost:${PORT}/v1/chat/completions"
    echo ""
    echo "To stop: docker stop ${CONTAINER_NAME}"
    exit 0
fi

# Stop old NIM containers that might be on the same port
docker stop nim-llama nim-llama-70b 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Create cache directory
mkdir -p "$HOME/.cache/huggingface/"

echo ""
echo "[1/3] Pulling TRT-LLM container (first time may take a while)..."
docker pull "$DOCKER_IMAGE"

echo ""
echo "[2/3] Starting Llama 3.3 70B (NVFP4) server..."
echo "  First run downloads ~35GB of model weights."
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN="$HF_TOKEN" \
    -e MODEL_HANDLE="$MODEL_HANDLE" \
    -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
    "$DOCKER_IMAGE" \
    bash -c "
        hf download $MODEL_HANDLE && \
        cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: auto
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
        trtllm-serve $MODEL_HANDLE \
            --max_batch_size 16 \
            --trust_remote_code \
            --port $PORT \
            --extra_llm_api_options /tmp/extra-llm-api-config.yml
    "

echo ""
echo "[3/3] Waiting for model to load (70B takes several minutes)..."
echo ""

MAX_WAIT=900  # 15 minutes for first-time download + engine build
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "=================================================="
        echo "LLAMA 3.3 70B READY!"
        echo "=================================================="
        echo ""
        echo "API endpoint: http://localhost:${PORT}/v1/chat/completions"
        echo "Model: ${MODEL_HANDLE}"
        echo ""
        echo "From other machines:"
        echo "  http://$(hostname -I | awk '{print $1}'):${PORT}"
        echo ""
        echo "Use with voice assistant:"
        echo "  ./voice_web.sh --nim-url http://localhost:${PORT}"
        echo ""
        echo "Test with:"
        echo "  curl -X POST http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"${MODEL_HANDLE}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
        echo ""
        echo "To view logs: docker logs -f ${CONTAINER_NAME}"
        echo "To stop:      docker stop ${CONTAINER_NAME}"
        exit 0
    fi
    
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  Waiting... ($WAITED seconds)"
        docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | grep -v "^$" | head -1 || true
    fi
    
    sleep 5
    WAITED=$((WAITED + 5))
done

echo ""
echo "WARNING: Server still initializing after ${MAX_WAIT}s"
echo "This is normal on first run (downloading + building TRT engine)"
echo ""
echo "Check progress with: docker logs -f ${CONTAINER_NAME}"
echo "Server is ready when you see 'Uvicorn running on'"
