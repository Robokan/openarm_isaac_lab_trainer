#!/usr/bin/env bash
set -e

# Bimanual Teleoperation / OpenPI Client for OpenArm (local)
# Controls both robot arms using Vive controllers, keyboard, or OpenPI policy server
#
# Usage:
#   # Teleoperation mode (default):
#   ./scripts/teleop_bimanual.sh                    # Use Vive controllers (requires SteamVR)
#   ./scripts/teleop_bimanual.sh --input keyboard  # Use keyboard for testing
#   ./scripts/teleop_bimanual.sh --input gamepad   # Use Xbox gamepad
#   ./scripts/teleop_bimanual.sh --checkpoint /path/to/model.pt  # Specify checkpoint
#
#   # OpenPI client mode (connect to π₀ policy server):
#   ./scripts/teleop_bimanual.sh --client                         # Connect to localhost:8000
#   ./scripts/teleop_bimanual.sh --client --host 192.168.1.100    # Connect to remote server
#   ./scripts/teleop_bimanual.sh --client --port 8080             # Use custom port
#   ./scripts/teleop_bimanual.sh --client --prompt "pick up the cube"  # Custom task prompt

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
echo "OPENARM BIMANUAL TELEOPERATION / OPENPI CLIENT"
echo "=========================================="
echo ""

# Check if running in client mode
CLIENT_MODE=false
for arg in "$@"; do
    if [[ "$arg" == "--client" ]]; then
        CLIENT_MODE=true
    fi
done

if [[ "$CLIENT_MODE" == "true" ]]; then
    echo "Mode: OpenPI Client (connecting to π₀ policy server)"
    echo ""
    echo "Options:"
    echo "  --host HOST     Policy server hostname (default: localhost)"
    echo "  --port PORT     Policy server port (default: 8000)"
    echo "  --prompt TEXT   Task instruction for VLA"
    echo ""
    echo "Make sure the policy server is running:"
    echo "  cd ../openpi && uv run scripts/serve_policy.py --env=ALOHA"
else
    echo "Mode: Teleoperation"
    echo ""
    echo "Controls:"
    echo "  - Vive Controllers: Move arms in real-time"
    echo "  - Triggers: Control grippers"
    echo ""
    echo "Keyboard fallback (--input keyboard):"
    echo "  - WASD/QE: Move active hand"
    echo "  - 1/2: Switch between left/right hand"
    echo "  - R: Reset poses"
    echo ""
    echo "Use --client flag to connect to OpenPI policy server instead"
fi
echo ""
echo "Starting..."
echo ""

# Install dependencies (pygame for gamepad, openvr for Vive)
python -m pip install pygame openvr >/dev/null 2>&1 || true

# Install openpi-client if running in client mode and not already installed
if [[ "$CLIENT_MODE" == "true" ]]; then
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
fi

# Run teleoperation/client script
python ./scripts/teleoperation/teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 ${EXTRA_ARGS} "$@"
