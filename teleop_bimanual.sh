#!/usr/bin/env bash
set -e

# Bimanual Teleoperation for OpenArm
# Controls both robot arms using Vive controllers or keyboard
#
# Usage:
#   ./teleop_bimanual.sh                    # Use Vive controllers (requires SteamVR)
#   ./teleop_bimanual.sh --input keyboard  # Use keyboard for testing
#   ./teleop_bimanual.sh --checkpoint /path/to/model.pt  # Specify checkpoint

CONTAINER_NAME="isaac-lab"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Find the latest bimanual checkpoint if not specified
CHECKPOINT=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        # User is specifying checkpoint, don't auto-find
        CHECKPOINT="user_specified"
    fi
done

if [[ -z "$CHECKPOINT" ]]; then
    # Try to find the latest bimanual checkpoint
    LATEST=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/openarm_isaac_lab/logs/rsl_rl -name 'model_*.pt' 2>/dev/null | grep -E '(bimanual|bi_reach|openarm_bi)' | sort -V | tail -1" 2>/dev/null)
    
    if [[ -n "$LATEST" ]]; then
        echo "Using latest bimanual checkpoint: $LATEST"
        EXTRA_ARGS="--checkpoint $LATEST"
    else
        echo "=========================================="
        echo "WARNING: No trained bimanual model found!"
        echo "=========================================="
        echo ""
        echo "Please train a bimanual reach model first:"
        echo "  ./train_bimanual_reach.sh --headless"
        echo ""
        echo "Or specify a checkpoint manually:"
        echo "  ./teleop_bimanual.sh --checkpoint /path/to/model.pt"
        echo ""
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "OPENARM BIMANUAL TELEOPERATION"
echo "=========================================="
echo ""
echo "Controls:"
echo "  - Vive Controllers: Move arms in real-time"
echo "  - Triggers: Control grippers"
echo ""
echo "Keyboard fallback (--input keyboard):"
echo "  - WASD/QE: Move active hand"
echo "  - TAB: Switch between left/right hand"
echo "  - R: Reset poses"
echo ""
echo "Starting..."
echo ""

# Install dependencies (pygame for gamepad, openvr for Vive)
docker exec ${CONTAINER_NAME} bash -c "/workspace/isaaclab/isaaclab.sh -p -m pip install pygame openvr 2>/dev/null || true"

# Run teleoperation script
docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 ${EXTRA_ARGS} $*"
