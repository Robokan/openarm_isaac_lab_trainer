#!/usr/bin/env bash
set -e

# Train OpenArm Bimanual Lift Task (local)
# Usage: ./scripts/train_bimanual_lift.sh [--headless] [--num_envs N] [additional args...]
# Examples:
#   ./scripts/train_bimanual_lift.sh --headless --num_envs 10   # Quick test with 10 envs
#   ./scripts/train_bimanual_lift.sh --headless                  # Full training

TASK="Isaac-Lift-Cube-OpenArm-Bi-v0"
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


echo "Training: ${TASK}"
echo "Args: $@"
echo ""

python ./scripts/reinforcement_learning/rsl_rl/train.py --task ${TASK} "$@"
