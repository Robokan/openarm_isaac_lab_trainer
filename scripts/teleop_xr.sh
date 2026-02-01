#!/usr/bin/env bash
set -e

# XR Teleoperation for OpenArm using WiVRn + Isaac Sim's built-in XR (local)
#
# Prerequisites:
#   1. WiVRn server running: flatpak run io.github.wivrn.wivrn
#   2. Vive XR Elite connected to WiVRn
#
# Usage:
#   ./scripts/teleop_xr.sh              # Bimanual XR teleoperation
#   ./scripts/teleop_xr.sh --keyboard   # Test with keyboard (no XR)

INPUT_MODE="xr"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check for keyboard flag
if [[ "$1" == "--keyboard" ]]; then
    INPUT_MODE="keyboard"
    shift
fi

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


# Check if WiVRn server is running (only for XR mode)
if [[ "$INPUT_MODE" == "xr" ]] && ! pgrep -f "wivrn" > /dev/null; then
    echo ""
    echo "WARNING: WiVRn server doesn't appear to be running!"
    echo "Start it with: flatpak run io.github.wivrn.wivrn"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

TASK="Isaac-Reach-OpenArm-Bi-v0"
LOG_PATH="openarm_bi_reach"

# Find latest checkpoint
CHECKPOINT=$(find "${REPO_ROOT}/logs/rsl_rl/${LOG_PATH}" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1)

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: No trained model found."
    echo "Train first with: ./scripts/train_bimanual_reach.sh --headless"
    exit 1
fi

echo ""
echo "=========================================="
echo "OPENARM XR TELEOPERATION"
echo "=========================================="
echo "Task: ${TASK}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Input Mode: ${INPUT_MODE}"
echo ""
if [[ "$INPUT_MODE" == "xr" ]]; then
    echo "Using VR controller tracking"
    echo "Make sure:"
    echo "  1. WiVRn server is running"
    echo "  2. Vive XR Elite is connected"
else
    echo "Using keyboard input (test mode)"
fi
echo "=========================================="
echo ""

# Run OpenArm teleop with XR
RUNTIME_JSON_PATH="${REPO_ROOT}/.openxr_runtime.json"
WIVRN_RUNTIME_JSON_DEFAULT="/var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/07b70b9a85dd76c10b6e240f9f84212c63beeaa4213dccabb743bbd82fe992e2/files/share/openxr/1/openxr_wivrn.json"
WIVRN_RUNTIME_LIB_DEFAULT="/var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/07b70b9a85dd76c10b6e240f9f84212c63beeaa4213dccabb743bbd82fe992e2/files/lib/wivrn/libopenxr_wivrn.so"
WIVRN_MONADO_LIB_DEFAULT="/var/lib/flatpak/app/io.github.wivrn.wivrn/x86_64/stable/07b70b9a85dd76c10b6e240f9f84212c63beeaa4213dccabb743bbd82fe992e2/files/lib/wivrn/libmonado_wivrn.so"
WIVRN_RUNTIME_JSON="${WIVRN_RUNTIME_JSON:-$WIVRN_RUNTIME_JSON_DEFAULT}"
WIVRN_RUNTIME_LIB="${WIVRN_RUNTIME_LIB:-$WIVRN_RUNTIME_LIB_DEFAULT}"
WIVRN_MONADO_LIB="${WIVRN_MONADO_LIB:-$WIVRN_MONADO_LIB_DEFAULT}"

if [[ ! -f "${WIVRN_RUNTIME_JSON}" ]]; then
    echo "Warning: WiVRn OpenXR runtime JSON not found."
    echo "Expected: ${WIVRN_RUNTIME_JSON}"
    echo "Falling back to a generated runtime JSON."
else
    export XR_RUNTIME_JSON="${WIVRN_RUNTIME_JSON}"
fi

if [[ ! -f "${WIVRN_RUNTIME_LIB}" ]]; then
    echo "Error: WiVRn OpenXR runtime library not found."
    echo "Expected: ${WIVRN_RUNTIME_LIB}"
    echo "Set WIVRN_RUNTIME_LIB to override."
    exit 1
fi

# Ensure WiVRn runtime dependencies are discoverable
if [[ "${INPUT_MODE}" == "xr" ]]; then
    WIVRN_LIB_DIR="$(dirname "${WIVRN_RUNTIME_LIB}")"
    WIVRN_PARENT_LIB_DIR="$(dirname "${WIVRN_LIB_DIR}")"
    if [[ -d "${WIVRN_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${WIVRN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
    if [[ -d "${WIVRN_PARENT_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${WIVRN_PARENT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
fi

# Validate that the OpenXR runtime library can be loaded
python - <<PY
import ctypes
import os
import sys

runtime = os.environ.get("WIVRN_RUNTIME_LIB", "${WIVRN_RUNTIME_LIB}")
try:
    ctypes.CDLL(runtime)
except OSError as exc:
    print("[ERROR] Failed to load WiVRn OpenXR runtime:", runtime, file=sys.stderr)
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)
PY

if [[ -z "${XR_RUNTIME_JSON:-}" ]]; then
    cat > "${RUNTIME_JSON_PATH}" <<EOF
{
  "file_format_version": "1.0.0",
  "runtime": {
    "name": "WiVRn",
    "library_path": "${WIVRN_RUNTIME_LIB}",
    "MND_libmonado_path": "${WIVRN_MONADO_LIB}"
  }
}
EOF
    export XR_RUNTIME_JSON="${RUNTIME_JSON_PATH}"
fi
KIT_ARGS=""
if [[ "${INPUT_MODE}" == "xr" ]]; then
    KIT_ARGS="--/persistent/xr/system/openxr/runtime=custom \
--/persistent/xr/system/openxr/activeRuntimeJSON=${XR_RUNTIME_JSON} \
--/app/extensions/enabled/omni.kit.xr.system.openxr=true \
--/app/extensions/enabled/omni.kit.xr.profile.vr=true \
--/app/extensions/enabled/omni.kit.xr.profile.ar=true"
fi

python ./scripts/teleoperation/teleop_bimanual.py \
    --task ${TASK} \
    --checkpoint ${CHECKPOINT} \
    --input ${INPUT_MODE} \
    --num_envs 1 \
    ${KIT_ARGS:+--kit_args "${KIT_ARGS}"} \
    "$@"
