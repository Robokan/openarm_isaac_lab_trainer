#!/usr/bin/env bash
set -e

# Play OpenArm Bimanual Lift Task
# Usage: ./play_bimanual_lift.sh [--checkpoint PATH] [additional args...]

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Lift-Cube-OpenArm-Bi-Play-v0"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Find latest checkpoint if not specified
CHECKPOINT_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        CHECKPOINT_ARG="provided"
        break
    fi
done

EXTRA_ARGS=""
if [[ -z "$CHECKPOINT_ARG" ]]; then
    # Find latest bimanual lift checkpoint
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    LATEST_DIR=$(ls -td "${REPO_ROOT}/logs/rsl_rl/openarm_bi_lift"/*/ 2>/dev/null | head -1)
    if [[ -n "$LATEST_DIR" ]]; then
        LATEST_MODEL=$(ls -t "${LATEST_DIR}"model_*.pt 2>/dev/null | head -1)
        if [[ -n "$LATEST_MODEL" ]]; then
            # Convert to container path
            CONTAINER_PATH="${LATEST_MODEL/${REPO_ROOT}/\/workspace\/openarm_isaac_lab}"
            echo "Using latest checkpoint: ${CONTAINER_PATH}"
            EXTRA_ARGS="--checkpoint ${CONTAINER_PATH}"
        fi
    fi
fi

echo "Playing: ${TASK}"
echo "Args: $@ ${EXTRA_ARGS}"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/play.py --task ${TASK} $* ${EXTRA_ARGS}"
