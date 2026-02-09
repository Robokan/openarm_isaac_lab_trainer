#!/usr/bin/env bash
set -e

# Play OpenArm Bimanual Reach Task
# Usage: ./play_bimanual_reach.sh [--checkpoint /path/to/model.pt] [--num_envs 16] [additional args...]

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Reach-OpenArm-Bi-v0"

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
    LATEST_CHECKPOINT=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/sparkpack/openarm_isaac_lab_trainer/logs/rsl_rl -name 'model_*.pt' -path '*bimanual*' -o -name 'model_*.pt' -path '*bi_reach*' 2>/dev/null | sort -V | tail -1")
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using latest checkpoint: ${LATEST_CHECKPOINT}"
        EXTRA_ARGS="${EXTRA_ARGS} --checkpoint ${LATEST_CHECKPOINT}"
    else
        echo "Warning: No checkpoint found. Train a model first with ./train_bimanual_reach.sh"
        exit 1
    fi
fi

# Ensure openarm package is installed
docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/_isaac_sim/python.sh -c 'import openarm' 2>/dev/null || { echo '[INFO] Installing OpenArm package...'; /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/sparkpack/openarm_isaac_lab_trainer/source/openarm; }"

echo "Playing: ${TASK}"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/sparkpack/openarm_isaac_lab_trainer && /workspace/isaaclab/isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/play.py --task ${TASK} ${EXTRA_ARGS} $*"
