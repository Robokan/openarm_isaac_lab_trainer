#!/usr/bin/env bash
set -e

# Unimanual Teleoperation for OpenArm (Single Arm)
# Controls one robot arm using a Vive controller or keyboard
#
# Usage:
#   ./teleop_unimanual.sh                    # Use Vive controller (requires SteamVR)
#   ./teleop_unimanual.sh --input keyboard  # Use keyboard for testing

CONTAINER_NAME="isaac-lab"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Find the latest unimanual reach checkpoint if not specified
CHECKPOINT=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        CHECKPOINT="user_specified"
    fi
done

if [[ -z "$CHECKPOINT" ]]; then
    LATEST=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/openarm_isaac_lab/logs/rsl_rl -name 'model_*.pt' -path '*openarm_reach*' 2>/dev/null | sort -V | tail -1" 2>/dev/null)
    
    if [[ -n "$LATEST" ]]; then
        echo "Using latest unimanual checkpoint: $LATEST"
        EXTRA_ARGS="--checkpoint $LATEST"
    else
        echo "WARNING: No trained unimanual model found!"
        echo "Train first with: ./train_reach.sh --headless"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "OPENARM UNIMANUAL TELEOPERATION"
echo "=========================================="
echo ""

# Install openvr if needed
docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/isaaclab.sh -p -m pip install openvr 2>/dev/null || true"

# Run teleoperation script
docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/teleop_unimanual.py --task Isaac-Reach-OpenArm-v0 ${EXTRA_ARGS} $*"
