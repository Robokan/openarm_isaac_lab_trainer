#!/usr/bin/env bash
set -e

# Bimanual IK Teleoperation for OpenArm (local)
# Controls both robot arms using keyboard or VR controllers with inverse kinematics
#
# Usage:
#   ./scripts/teleop_bimanual.sh --input keyboard    # Keyboard control
#   ./scripts/teleop_bimanual.sh --input xr           # VR handtracking (requires WiVRn)
#   ./scripts/teleop_bimanual.sh --input vive          # Vive controllers (requires SteamVR)
#   ./scripts/teleop_bimanual.sh --input gamepad       # Xbox gamepad

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

# Avoid system CUDA libs overriding Isaac Sim bundled libs
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _clean_ld=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda' | paste -sd: -)
    if [[ -n "${_clean_ld}" ]]; then
        export LD_LIBRARY_PATH="${_clean_ld}"
    else
        unset LD_LIBRARY_PATH
    fi
fi

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

# Detect input mode from arguments
INPUT_MODE="keyboard"
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

# Install optional dependencies based on input mode (suppress errors)
if [[ "$INPUT_MODE" == "gamepad" ]]; then
    python -m pip install pygame >/dev/null 2>&1 || true
elif [[ "$INPUT_MODE" == "vive" ]]; then
    python -m pip install openvr >/dev/null 2>&1 || true
fi

# Run IK teleoperation script
python ./scripts/teleoperation/teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 "$@"
