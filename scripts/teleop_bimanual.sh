#!/usr/bin/env bash
set -e

# Bimanual Teleoperation for OpenArm (local)
# Controls both robot arms using Vive controllers or keyboard
#
# Usage:
#   ./scripts/teleop_bimanual.sh                    # Use Vive controllers (requires SteamVR)
#   ./scripts/teleop_bimanual.sh --input keyboard  # Use keyboard for testing
#   ./scripts/teleop_bimanual.sh --checkpoint /path/to/model.pt  # Specify checkpoint

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


# Find the latest bimanual checkpoint if not specified
CHECKPOINT=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        CHECKPOINT="user_specified"
    fi
done

if [[ -z "$CHECKPOINT" ]]; then
    LATEST=$(find "${REPO_ROOT}/logs/rsl_rl" -name 'model_*.pt' 2>/dev/null | grep -E '(bimanual|bi_reach|openarm_bi)' | sort -V | tail -1)

    if [[ -n "$LATEST" ]]; then
        echo "Using latest bimanual checkpoint: $LATEST"
        EXTRA_ARGS="--checkpoint $LATEST"
    else
        echo "=========================================="
        echo "WARNING: No trained bimanual model found!"
        echo "=========================================="
        echo ""
        echo "Please train a bimanual reach model first:"
        echo "  ./scripts/train_bimanual_reach.sh --headless"
        echo ""
        echo "Or specify a checkpoint manually:"
        echo "  ./scripts/teleop_bimanual.sh --checkpoint /path/to/model.pt"
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
python -m pip install pygame openvr >/dev/null 2>&1 || true

# Run teleoperation script
python ./scripts/teleoperation/teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 ${EXTRA_ARGS} "$@"
