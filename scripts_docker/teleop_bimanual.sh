#!/usr/bin/env bash
set -e

# Bimanual IK Teleoperation for OpenArm (Docker)
# Controls both robot arms using keyboard or VR controllers with inverse kinematics
#
# Usage:
#   ./teleop_bimanual.sh --input keyboard    # Keyboard control
#   ./teleop_bimanual.sh --input xr           # VR handtracking (requires WiVRn)
#   ./teleop_bimanual.sh --input vive          # Vive controllers (requires SteamVR)
#   ./teleop_bimanual.sh --input gamepad       # Xbox gamepad

CONTAINER_NAME="isaac-lab"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Detect input mode from arguments
INPUT_MODE="keyboard"
prev_arg=""
for i in "$@"; do
    if [[ "$prev_arg" == "--input" ]]; then
        INPUT_MODE="$i"
    fi
    prev_arg="$i"
done

echo ""
echo "=========================================="
echo "OPENARM BIMANUAL IK TELEOPERATION"
echo "=========================================="
echo ""
echo "Input mode: ${INPUT_MODE}"
echo ""

if [[ "$INPUT_MODE" == "keyboard" ]]; then
    echo "Keyboard Controls:"
    echo "  Position:  W/S (X), A/D (Y), Q/E (Z)"
    echo "  Rotation:  I/K (pitch), J/L (yaw), U/O (roll)"
    echo "  Gripper:   ; (close), ' (open)"
    echo "  Hand:      1 (left), 2 (right)"
    echo "  Other:     C (spawn cube), M (toggle markers), R (reset)"
elif [[ "$INPUT_MODE" == "xr" ]]; then
    echo "Using VR handtracking (XR)"
    echo "Make sure WiVRn server is running"
elif [[ "$INPUT_MODE" == "vive" ]]; then
    echo "Using Vive controllers"
    echo "Make sure SteamVR is running"
elif [[ "$INPUT_MODE" == "gamepad" ]]; then
    echo "Using Xbox/gamepad controller"
fi
echo ""
echo "Starting..."
echo ""

# Ensure openarm package is installed
docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/_isaac_sim/python.sh -c 'import openarm' 2>/dev/null || { echo '[INFO] Installing OpenArm package...'; /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/sparkpack/openarm_isaac_lab_trainer/source/openarm; }"

# Install optional dependencies based on input mode
if [[ "$INPUT_MODE" == "gamepad" ]]; then
    docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/isaaclab.sh -p -m pip install pygame 2>/dev/null || true"
elif [[ "$INPUT_MODE" == "vive" ]]; then
    docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/isaaclab.sh -p -m pip install openvr 2>/dev/null || true"
fi

# Run IK teleoperation script
docker exec -it ${CONTAINER_NAME} bash -c "cd /workspace/sparkpack/openarm_isaac_lab_trainer && /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 $*"
