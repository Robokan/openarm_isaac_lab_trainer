#!/usr/bin/env bash
set -e

# XR Teleoperation for OpenArm using WiVRn + Isaac Sim's built-in XR (local)
#
# Prerequisites:
#   1. WiVRn server running: flatpak run io.github.wivrn.wivrn
#   2. Vive XR Elite connected to WiVRn
#
# Usage:
#   ./scripts/teleop_xr.sh              # Bimanual XR teleoperation
#   ./scripts/teleop_xr.sh --keyboard   # Test with keyboard (no XR)

INPUT_MODE="xr"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check for keyboard flag
if [[ "$1" == "--keyboard" ]]; then
    INPUT_MODE="keyboard"
    shift
fi

cd "${REPO_ROOT}"

# Ensure OpenArm package is installed in the active environment
if python - <<'PY_OPENARM_CHECK'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("openarm") else 1)
PY_OPENARM_CHECK
then
    :
else
    echo "[INFO] Installing OpenArm package (editable)..."
    python -m pip install -e "${REPO_ROOT}/source/openarm"
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
CHECKPOINT=$(find "${REPO_ROOT}/logs/rsl_rl/${LOG_PATH}" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1)

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: No trained model found."
    echo "Train first with: ./scripts/train_bimanual_reach.sh --headless"
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
export XR_RUNTIME_JSON=/root/.config/openxr/1/active_runtime.json
python ./scripts/teleoperation/teleop_bimanual.py \
    --task ${TASK} \
    --checkpoint ${CHECKPOINT} \
    --input ${INPUT_MODE} \
    --num_envs 1 \
    "$@"
