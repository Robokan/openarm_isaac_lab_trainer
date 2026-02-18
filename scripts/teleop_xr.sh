#!/usr/bin/env bash
set -e

# IK Teleoperation for OpenArm using WiVRn + Isaac Sim's built-in XR
#
# Uses inverse kinematics for direct control - no trained model needed.
#
# Prerequisites:
#   1. WiVRn server running: flatpak run io.github.wivrn.wivrn
#   2. Vive XR Elite connected to WiVRn
#
# Usage:
#   ./scripts/teleop_xr.sh              # XR teleoperation with VR controllers
#   ./scripts/teleop_xr.sh --keyboard   # Test with keyboard (no XR)

INPUT_MODE="xr"
SCRIPT_MODE=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRA_ARGS=()

# Check for flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keyboard)
            INPUT_MODE="keyboard"
            shift
            ;;
        --script)
            SCRIPT_MODE="$2"
            EXTRA_ARGS+=("--script" "$2")
            shift 2
            ;;
        --script=*)
            SCRIPT_MODE="${1#*=}"
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If running a script, use keyboard mode (no VR needed)
if [[ -n "${SCRIPT_MODE}" ]]; then
    INPUT_MODE="keyboard"
fi

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


# Check if WiVRn server is running (only for XR mode, not for script mode)
if [[ "$INPUT_MODE" == "xr" ]] && [[ -z "${SCRIPT_MODE}" ]] && ! pgrep -f "wivrn" > /dev/null; then
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

TASK="Isaac-Reach-OpenArm-Bi-Teleop-v0"

echo ""
echo "=========================================="
echo "OPENARM IK TELEOPERATION"
echo "=========================================="
echo "Task: ${TASK}"
echo "Control Mode: Inverse Kinematics (IK)"
if [[ -n "${SCRIPT_MODE}" ]]; then
    echo "Input Mode: Script"
    echo "Script: ${SCRIPT_MODE}"
elif [[ "$INPUT_MODE" == "xr" ]]; then
    echo "Input Mode: VR Controllers"
    echo "Make sure:"
    echo "  1. WiVRn server is running"
    echo "  2. Vive XR Elite is connected"
else
    echo "Input Mode: Keyboard (test mode)"
fi
echo "=========================================="
echo ""

# XR/VR setup (only needed for xr input mode)
KIT_ARGS=""
if [[ "${INPUT_MODE}" == "xr" ]]; then
    RUNTIME_JSON_PATH="${REPO_ROOT}/.openxr_runtime.json"
    
    # Dynamically find WiVRn Flatpak installation (path changes with updates)
    WIVRN_RUNTIME_LIB_DEFAULT="$(find /var/lib/flatpak/app/io.github.wivrn.wivrn -name 'libopenxr_wivrn.so' 2>/dev/null | head -1)"
    if [[ -n "${WIVRN_RUNTIME_LIB_DEFAULT}" ]]; then
        WIVRN_LIB_DIR="$(dirname "${WIVRN_RUNTIME_LIB_DEFAULT}")"
        WIVRN_FILES_DIR="$(dirname "$(dirname "${WIVRN_LIB_DIR}")")"
        WIVRN_RUNTIME_JSON_DEFAULT="${WIVRN_FILES_DIR}/share/openxr/1/openxr_wivrn.json"
        WIVRN_MONADO_LIB_DEFAULT="${WIVRN_LIB_DIR}/libmonado_wivrn.so"
    else
        WIVRN_RUNTIME_JSON_DEFAULT=""
        WIVRN_MONADO_LIB_DEFAULT=""
    fi
    
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
    WIVRN_LIB_DIR="$(dirname "${WIVRN_RUNTIME_LIB}")"
    WIVRN_PARENT_LIB_DIR="$(dirname "${WIVRN_LIB_DIR}")"
    if [[ -d "${WIVRN_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${WIVRN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
    if [[ -d "${WIVRN_PARENT_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${WIVRN_PARENT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
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

    XR_RENDER_WIDTH="${XR_RENDER_WIDTH:-1280}"
    XR_RENDER_HEIGHT="${XR_RENDER_HEIGHT:-720}"
    KIT_ARGS="--/persistent/xr/system/openxr/runtime=custom \
--/persistent/xr/system/openxr/activeRuntimeJSON=${XR_RUNTIME_JSON} \
--/app/extensions/enabled/omni.kit.xr.system.openxr=true \
--/app/extensions/enabled/omni.kit.xr.profile.vr=true \
--/app/extensions/enabled/omni.kit.xr.profile.ar=true \
--/app/renderer/resolution/width=${XR_RENDER_WIDTH} \
--/app/renderer/resolution/height=${XR_RENDER_HEIGHT} \
--/app/window/width=${XR_RENDER_WIDTH} \
--/app/window/height=${XR_RENDER_HEIGHT}"
fi

# Build command
TELEOP_CMD="python ./scripts/teleoperation/teleop_bimanual.py \
    --task ${TASK} \
    --input ${INPUT_MODE} \
    --num_envs 1"

if [[ -n "${KIT_ARGS}" ]]; then
    TELEOP_CMD="${TELEOP_CMD} --kit_args \"${KIT_ARGS}\""
fi

# Execute the command with any extra args
eval ${TELEOP_CMD} "${EXTRA_ARGS[@]}"
