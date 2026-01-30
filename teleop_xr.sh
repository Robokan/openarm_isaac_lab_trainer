#!/usr/bin/env bash
set -e

# XR Teleoperation for OpenArm using WiVRn + Isaac Sim's built-in XR
#
# Prerequisites:
#   1. WiVRn server running: flatpak run io.github.wivrn.wivrn
#   2. Vive XR Elite connected to WiVRn
#   3. Container running: ./start_container.sh
#
# Usage:
#   ./teleop_xr.sh              # Bimanual XR teleoperation
#   ./teleop_xr.sh --keyboard   # Test with keyboard (no XR)

CONTAINER_NAME="isaac-lab"
INPUT_MODE="xr"

# Check for keyboard flag
if [[ "$1" == "--keyboard" ]]; then
    INPUT_MODE="keyboard"
    shift
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Check if WiVRn server is running (only for XR mode)
if [[ "$INPUT_MODE" == "xr" ]] && ! pgrep -f "wivrn" > /dev/null; then
    echo ""
    echo "WARNING: WiVRn server doesn't appear to be running!"
    echo "Start it with: flatpak run io.github.wivrn.wivrn"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

TASK="Isaac-Reach-OpenArm-Bi-v0"
LOG_PATH="openarm_bi_reach"

# Find latest checkpoint
CHECKPOINT=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/openarm_isaac_lab/logs/rsl_rl/${LOG_PATH} -name 'model_*.pt' 2>/dev/null | sort -V | tail -1" 2>/dev/null)

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: No trained model found."
    echo "Train first with: ./train_bimanual_reach.sh --headless"
    exit 1
fi

echo ""
echo "=========================================="
echo "OPENARM XR TELEOPERATION"
echo "=========================================="
echo "Task: ${TASK}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Input Mode: ${INPUT_MODE}"
echo ""
if [[ "$INPUT_MODE" == "xr" ]]; then
    echo "Using VR controller tracking"
    echo "Make sure:"
    echo "  1. WiVRn server is running"
    echo "  2. Vive XR Elite is connected"
else
    echo "Using keyboard input (test mode)"
fi
echo "=========================================="
echo ""

# Run OpenArm teleop with XR
docker exec -it ${CONTAINER_NAME} bash -c "
    export XR_RUNTIME_JSON=/root/.config/openxr/1/active_runtime.json
    cd /workspace/openarm_isaac_lab
    /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/teleop_bimanual.py \
        --task ${TASK} \
        --checkpoint ${CHECKPOINT} \
        --input ${INPUT_MODE} \
        --num_envs 1 \
        $@
"
