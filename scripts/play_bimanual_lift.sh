#!/usr/bin/env bash
set -e

# Play OpenArm Bimanual Lift Task (local)
# Usage: ./scripts/play_bimanual_lift.sh [--checkpoint PATH] [additional args...]

TASK="Isaac-Lift-Cube-OpenArm-Bi-Play-v0"
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

# Find latest checkpoint if not specified
CHECKPOINT_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        CHECKPOINT_ARG="provided"
        break
    fi
done

if [[ -z "$CHECKPOINT_ARG" ]]; then
    # Find latest bimanual lift checkpoint
    LATEST_DIR=$(ls -td "${REPO_ROOT}/logs/rsl_rl/openarm_bi_lift"/*/ 2>/dev/null | head -1)
    if [[ -n "$LATEST_DIR" ]]; then
        LATEST_MODEL=$(ls -t "${LATEST_DIR}"model_*.pt 2>/dev/null | head -1)
        if [[ -n "$LATEST_MODEL" ]]; then
            echo "Using latest checkpoint: ${LATEST_MODEL}"
            set -- "$@" --checkpoint "${LATEST_MODEL}"
        fi
    fi
fi

echo "Playing: ${TASK}"
echo "Args: $@"
echo ""

python ./scripts/reinforcement_learning/rsl_rl/play.py --task ${TASK} "$@"
