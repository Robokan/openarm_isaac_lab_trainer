#!/usr/bin/env bash
set -e

# Train OpenArm Bimanual Reach Task (local)
# Usage: ./scripts/train_bimanual_reach.sh [--headless] [additional args...]

TASK="Isaac-Reach-OpenArm-Bi-v0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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


echo "Training: ${TASK}"
echo "Args: $@"
echo ""

python ./scripts/reinforcement_learning/rsl_rl/train.py --task ${TASK} "$@"
