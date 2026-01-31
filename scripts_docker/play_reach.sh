#!/usr/bin/env bash
set -e

# Play OpenArm Unimanual Reach Task
# Usage: ./play_reach.sh [--checkpoint /path/to/model.pt] [--num_envs 16] [additional args...]

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Reach-OpenArm-v0"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Default args
EXTRA_ARGS="--num_envs 16"

# If no checkpoint specified, try to find the latest one
if [[ ! "$*" =~ "--checkpoint" ]]; then
    LATEST_CHECKPOINT=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/openarm_isaac_lab/logs/rsl_rl -name 'model_*.pt' -path '*openarm_reach*' 2>/dev/null | sort -V | tail -1")
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using latest checkpoint: ${LATEST_CHECKPOINT}"
        EXTRA_ARGS="${EXTRA_ARGS} --checkpoint ${LATEST_CHECKPOINT}"
    else
        echo "Warning: No checkpoint found. Train a model first with ./train_reach.sh"
        exit 1
    fi
fi

echo "Playing: ${TASK}"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/play.py --task ${TASK} ${EXTRA_ARGS} $*"
