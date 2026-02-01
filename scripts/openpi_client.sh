#!/usr/bin/env bash
set -e

# OpenPI Client for Bimanual OpenArm (local)
# Connects to π₀ policy server and executes VLA actions in Isaac Lab simulation
#
# Usage:
#   ./scripts/openpi_client.sh                              # Connect to localhost:8000
#   ./scripts/openpi_client.sh --host 192.168.1.100         # Connect to remote server
#   ./scripts/openpi_client.sh --port 8080                  # Use custom port
#   ./scripts/openpi_client.sh --prompt "pick up the cube"  # Custom task prompt
#   ./scripts/openpi_client.sh --num_episodes 5             # Run 5 episodes
#   ./scripts/openpi_client.sh --checkpoint /path/to/model.pt  # Specify checkpoint

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
        echo "  ./scripts/openpi_client.sh --checkpoint /path/to/model.pt"
        echo ""
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "OPENARM BIMANUAL - OPENPI CLIENT"
echo "=========================================="
echo ""
echo "Connects to π₀ policy server and executes VLA joint commands"
echo ""
echo "Options:"
echo "  --host HOST           Policy server hostname (default: localhost)"
echo "  --port PORT           Policy server port (default: 8000)"
echo "  --prompt TEXT         Task instruction for VLA"
echo "  --action_horizon N    Action chunk size (default: 10)"
echo "  --max_hz HZ           Max control frequency (default: 50)"
echo "  --num_episodes N      Number of episodes (default: 1)"
echo "  --max_episode_steps N Max steps per episode (default: 1000)"
echo ""
echo "Make sure the policy server is running:"
echo "  cd ../openpi && uv run scripts/serve_policy.py --env=ALOHA"
echo ""
echo "Starting..."
echo ""

# Install openpi-client if not already installed
if ! python -c "import openpi_client" 2>/dev/null; then
    OPENPI_CLIENT_PATH="${REPO_ROOT}/../openpi/packages/openpi-client"
    if [[ -d "$OPENPI_CLIENT_PATH" ]]; then
        echo "[INFO] Installing openpi-client package..."
        python -m pip install -e "$OPENPI_CLIENT_PATH" >/dev/null 2>&1 || {
            echo "[WARN] Failed to install openpi-client. Install manually:"
            echo "  pip install -e $OPENPI_CLIENT_PATH"
        }
    else
        echo "[WARN] openpi-client not found at $OPENPI_CLIENT_PATH"
        echo "[INFO] Install manually: pip install -e /path/to/openpi/packages/openpi-client"
    fi
fi

# Run OpenPI client script
python ./scripts/teleoperation/openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 ${EXTRA_ARGS} "$@"
