#!/usr/bin/env bash
set -e

# Train OpenArm Bimanual Reach Task
# Usage: ./train_bimanual_reach.sh [--headless] [additional args...]

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Reach-OpenArm-Bi-v0"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

echo "Training: ${TASK}"
echo "Args: $@"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py --task ${TASK} $*"
