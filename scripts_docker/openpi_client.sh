#!/usr/bin/env bash
set -e

# OpenPI Client for Bimanual OpenArm (Docker version)
# Connects to π₀ policy server and executes VLA actions in Isaac Lab simulation
#
# Prerequisites:
#   1. Container running: ./start_container.sh
#   2. OpenPI policy server running on host (localhost:8000)
#
# Usage:
#   ./openpi_client.sh                                    # Connect to localhost:8000
#   ./openpi_client.sh --host 192.168.1.100              # Connect to remote server
#   ./openpi_client.sh --prompt "pick up the cube"       # Custom task prompt
#   ./openpi_client.sh --interactive                     # Interactive prompt mode
#   ./openpi_client.sh --spawn-objects                   # Spawn random object on workspace
#   ./openpi_client.sh --headless                        # Run without GUI (no X11 needed)

CONTAINER_NAME="isaac-lab"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

TASK="Isaac-Reach-OpenArm-Bi-Teleop-v0"

# Parse arguments
HOST="localhost"
PORT="8000"
PROMPT="lift arms on the table."
INTERACTIVE=""
SPAWN_OBJECTS=""
HEADLESS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --interactive) INTERACTIVE="--interactive"; shift ;;
        --spawn-objects) SPAWN_OBJECTS="--spawn-objects"; shift ;;
        --headless) HEADLESS="--headless"; shift ;;
        *) shift ;;
    esac
done

echo ""
echo "=========================================="
echo "OPENARM BIMANUAL - OPENPI CLIENT (Docker)"
echo "=========================================="
echo "Task: ${TASK}"
echo "Host: ${HOST}:${PORT}"
echo "Prompt: ${PROMPT}"
if [ -n "${SPAWN_OBJECTS}" ]; then
    echo "Spawn Objects: enabled"
fi
if [ -n "${HEADLESS}" ]; then
    echo "Mode: headless (no GUI)"
fi
echo "=========================================="
echo ""

# Check/install packages only if needed (quiet check)
docker exec ${CONTAINER_NAME} bash -c "
    /workspace/isaaclab/_isaac_sim/python.sh -c 'import openarm' 2>/dev/null || \
        /workspace/isaaclab/_isaac_sim/python.sh -m pip install -q -e /workspace/sparkpack/openarm_isaac_lab_trainer/source/openarm
    /workspace/isaaclab/_isaac_sim/python.sh -c 'import openpi_client' 2>/dev/null || \
        /workspace/isaaclab/_isaac_sim/python.sh -m pip install -q -e /workspace/sparkpack/openpi/packages/openpi-client
" 2>/dev/null

# Run OpenPI client with properly escaped arguments
docker exec -it ${CONTAINER_NAME} bash -c "
    cd /workspace/sparkpack/openarm_isaac_lab_trainer
    /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/openpi_client_bimanual.py \
        --task '${TASK}' \
        --num_envs 1 \
        --host '${HOST}' \
        --port ${PORT} \
        --prompt '${PROMPT}' \
        ${INTERACTIVE} \
        ${SPAWN_OBJECTS} \
        ${HEADLESS}
"
