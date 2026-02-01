#!/usr/bin/env bash
set -e

# Unimanual Teleoperation for OpenArm (local)
# Controls one robot arm using a Vive controller or keyboard
#
# Usage:
#   ./scripts/teleop_unimanual.sh                    # Use Vive controller (requires SteamVR)
#   ./scripts/teleop_unimanual.sh --input keyboard  # Use keyboard for testing

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


# Find the latest unimanual reach checkpoint if not specified
CHECKPOINT=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        CHECKPOINT="user_specified"
    fi
done

if [[ -z "$CHECKPOINT" ]]; then
    LATEST=$(find "${REPO_ROOT}/logs/rsl_rl" -name 'model_*.pt' -path '*openarm_reach*' 2>/dev/null | sort -V | tail -1)

    if [[ -n "$LATEST" ]]; then
        echo "Using latest unimanual checkpoint: $LATEST"
        EXTRA_ARGS="--checkpoint $LATEST"
    else
        echo "WARNING: No trained unimanual model found!"
        echo "Train first with: ./scripts/train_reach.sh --headless"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "OPENARM UNIMANUAL TELEOPERATION"
echo "=========================================="
echo ""

# Install openvr if needed
python -m pip install openvr >/dev/null 2>&1 || true

# Run teleoperation script
python ./scripts/teleoperation/teleop_unimanual.py --task Isaac-Reach-OpenArm-v0 ${EXTRA_ARGS} "$@"
