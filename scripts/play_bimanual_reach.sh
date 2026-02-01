#!/usr/bin/env bash
set -e

# Play OpenArm Bimanual Reach Task (local)
# Usage: ./scripts/play_bimanual_reach.sh [--checkpoint /path/to/model.pt] [--num_envs 16] [additional args...]

TASK="Isaac-Reach-OpenArm-Bi-v0"
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


# Default args
EXTRA_ARGS="--num_envs 16"

# If no checkpoint specified, try to find the latest one
if [[ ! "$*" =~ "--checkpoint" ]]; then
    LATEST_CHECKPOINT=$(find "${REPO_ROOT}/logs/rsl_rl" -name 'model_*.pt' -path '*bimanual*' -o -name 'model_*.pt' -path '*bi_reach*' 2>/dev/null | sort -V | tail -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using latest checkpoint: ${LATEST_CHECKPOINT}"
        EXTRA_ARGS="${EXTRA_ARGS} --checkpoint ${LATEST_CHECKPOINT}"
    else
        echo "Warning: No checkpoint found. Train a model first with ./scripts/train_bimanual_reach.sh"
        exit 1
    fi
fi

echo "Playing: ${TASK}"
echo ""

python ./scripts/reinforcement_learning/rsl_rl/play.py --task ${TASK} ${EXTRA_ARGS} "$@"
