#!/usr/bin/env bash
set -e

# SparkJAX Digital Twin -- Isaac Sim bridge
#
# Launches the SparkPackDigialTwin USD scene in Isaac Sim and runs the
# station manager that advertises virtual CAN interfaces to SparkJAX.
# After launching, tell Zeus "I just added new robots" or "the sim is
# running" to start the setup wizard.
#
# Prerequisites:
#   1. ROS2 Humble sourced:  source /opt/ros/humble/setup.bash
#   2. SparkJAX workspace:   source <sparkjax_ws>/install/setup.bash
#   3. SparkJAX running:     ros2 launch sparkjax sparkjax.launch.py
#
# Usage:
#   ./scripts/sparkjax_digital_twin.sh
#   ./scripts/sparkjax_digital_twin.sh --headless     # No GUI
#   ./scripts/sparkjax_digital_twin.sh --gpu 0        # Specific GPU

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --gpu=*)
            GPU_ID="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo ""
echo "=========================================="
echo "SPARKJAX DIGITAL TWIN"
echo "=========================================="
echo "Station manager will:"
echo "  1. Load SparkPackDigialTwin USD scene"
echo "  2. Discover robot pairs"
echo "  3. Advertise virtual CAN interfaces"
echo "  4. Wait for wizard registration via Zeus"
echo "  5. Stream joint states to SparkJAX"
echo ""
echo "Make sure SparkJAX is running, then tell"
echo "Zeus: \"I just added new robots\""
echo "=========================================="
echo ""

# Auto-detect Isaac Sim bundled ROS2 bridge (for running inside Isaac Lab container)
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
ROS_BRIDGE="${ISAACLAB_PATH}/_isaac_sim/exts/isaacsim.ros2.bridge/humble"

if [[ -d "${ROS_BRIDGE}/rclpy" ]]; then
    echo "Using Isaac Sim bundled ROS2 Humble bridge"
    export PYTHONPATH="${ROS_BRIDGE}/rclpy:${PYTHONPATH}"
    export LD_LIBRARY_PATH="${ROS_BRIDGE}/lib:${LD_LIBRARY_PATH}"
elif command -v ros2 &> /dev/null; then
    echo "Using system ROS2"
else
    echo "ERROR: No ROS2 found. Either:"
    echo "  - Run inside Isaac Lab container (bundled ROS2 bridge)"
    echo "  - Or source ROS2: source /opt/ros/humble/setup.bash"
    exit 1
fi

# FastDDS UDP transport for cross-container communication
FASTDDS_CFG="${REPO_ROOT}/../SparkJAX/docker/fastdds_no_shm.xml"
if [[ -z "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]] && [[ -f "${FASTDDS_CFG}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_CFG}"
    echo "FastDDS UDP transport enabled"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

if [[ -n "${GPU_ID}" ]]; then
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_PREFERRED_DEVICE="${GPU_ID}"
    echo "Preferred GPU: ${GPU_ID}"
fi

cd "${REPO_ROOT}"

# Use Isaac Sim's Python if available
ISAAC_PYTHON="${ISAACLAB_PATH}/_isaac_sim/python.sh"
if [[ -x "${ISAAC_PYTHON}" ]]; then
    "${ISAAC_PYTHON}" ./scripts/sparkjax_bridge/station_manager.py "${EXTRA_ARGS[@]}"
else
    python3 ./scripts/sparkjax_bridge/station_manager.py "${EXTRA_ARGS[@]}"
fi
