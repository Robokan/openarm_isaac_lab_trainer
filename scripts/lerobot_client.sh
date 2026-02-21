#!/usr/bin/env bash
set -e

# LeRobot ACT Client for Bimanual OpenArm
# Runs a trained LeRobot ACT model directly in Isaac Lab simulation
#
# Usage:
#   ./scripts/lerobot_client.sh --checkpoint outputs/openarm_act/checkpoints/last/pretrained_model
#   ./scripts/lerobot_client.sh --checkpoint /path/to/checkpoint --num_episodes 5
#   ./scripts/lerobot_client.sh --checkpoint /path/to/checkpoint --auto-spawn 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

# Activate Isaac Lab environment
source "${REPO_ROOT}/scripts/source_me.sh"

# Avoid system CUDA libs overriding Isaac Sim bundled libs
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _clean_ld=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda' | paste -sd: -)
    if [[ -n "${_clean_ld}" ]]; then
        export LD_LIBRARY_PATH="${_clean_ld}"
    else
        unset LD_LIBRARY_PATH
    fi
fi

# Ensure OpenArm package is installed
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

# Check if LeRobot is installed
if ! python -c "import lerobot" 2>/dev/null; then
    echo "[ERROR] LeRobot not installed!"
    echo "[INFO] Install with: pip install lerobot"
    exit 1
fi

echo ""
echo "=========================================="
echo "OPENARM BIMANUAL - LEROBOT ACT CLIENT"
echo "=========================================="
echo ""
echo "Runs a trained LeRobot ACT model in Isaac Lab simulation"
echo ""
echo "Options:"
echo "  --checkpoint PATH     Path to LeRobot checkpoint (required)"
echo "  --num_episodes N      Number of episodes (default: 1)"
echo "  --max_episode_steps N Max steps per episode (default: 1000)"
echo "  --fps HZ              Control frequency (default: 50)"
echo ""
echo "Starting..."
echo ""

# Run LeRobot client script
python ./scripts/teleoperation/lerobot_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 "$@"
