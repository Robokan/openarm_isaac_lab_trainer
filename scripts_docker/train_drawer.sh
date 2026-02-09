#!/usr/bin/env bash
set -e

# Train OpenArm Open Drawer Task
# Usage: ./train_drawer.sh [--headless] [additional args...]

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Open-Drawer-OpenArm-v0"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Ensure openarm package is installed
docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/_isaac_sim/python.sh -c 'import openarm' 2>/dev/null || { echo '[INFO] Installing OpenArm package...'; /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/sparkpack/openarm_isaac_lab_trainer/source/openarm; }"

echo "Training: ${TASK}"
echo "Args: $@"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/sparkpack/openarm_isaac_lab_trainer && /workspace/isaaclab/isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py --task ${TASK} $*"
