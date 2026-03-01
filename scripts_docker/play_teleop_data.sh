#!/usr/bin/env bash
# Play back recorded LeRobot teleoperation data (Docker version)
#
# Usage:
#   ./scripts_docker/play_teleop_data.sh                    # Play all episodes
#   ./scripts_docker/play_teleop_data.sh --episode 5        # Play specific episode
#   ./scripts_docker/play_teleop_data.sh --episode 2-4      # Play range (2, 3, 4)
#   ./scripts_docker/play_teleop_data.sh --episode 1,3,6    # Play list (1, 3, 6)
#   ./scripts_docker/play_teleop_data.sh --loop             # Loop playback
#   ./scripts_docker/play_teleop_data.sh --verify           # Just verify data, no simulation
#   ./scripts_docker/play_teleop_data.sh --no-collect-video # Skip video capture during playback
#   ./scripts_docker/play_teleop_data.sh --label            # Prompt for task label after each episode
#   ./scripts_docker/play_teleop_data.sh --headless         # Run without GUI at max speed
#
# Controls during playback:
#   Q     - Quit
#   N     - Next episode
#   SPACE - Pause/Resume

set -e

CONTAINER_NAME="isaac-lab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Start it with: ./start_container.sh"
    exit 1
fi

# Data directory - check multiple locations (host paths mapped into container)
# Priority: 1) LeRobot format in repo, 2) raw teleop in repo, 3) home datasets dir
find_data_dir() {
    local candidates=(
        "/workspace/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data"
        "/workspace/sparkpack/vla_teleop_data"
    )
    for dir in "${candidates[@]}"; do
        # LeRobot native format (meta/info.json)
        if docker exec ${CONTAINER_NAME} test -f "${dir}/meta/info.json" 2>/dev/null; then
            echo "$dir"
            return
        fi
        # Fallback format (episodes/ directory)
        if docker exec ${CONTAINER_NAME} test -d "${dir}/episodes" 2>/dev/null; then
            echo "$dir"
            return
        fi
    done
    echo ""
}

# Ensure X11 forwarding is enabled (may have been lost after reboot)
xhost +local:docker >/dev/null 2>&1 || true
xhost +local:root >/dev/null 2>&1 || true

DATA_DIR="$(find_data_dir)"
if [ -z "$DATA_DIR" ]; then
    echo "Error: No teleop data found. Searched:"
    echo "  /workspace/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data"
    echo "  /workspace/sparkpack/vla_teleop_data"
    echo ""
    echo "Record some teleoperation data first with teleop_xr.sh"
    exit 1
fi

echo "Using data: ${DATA_DIR}"

# Build extra args string with proper quoting
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="${EXTRA_ARGS} $(printf '%q' "$arg")"
done

# Run the playback script
docker exec -it ${CONTAINER_NAME} bash -c "
    cd /workspace/isaaclab && source /workspace/isaaclab/_isaac_sim/setup_conda_env.sh
    
    # CUDA 13 libs needed for Blackwell/GB10 GPUs
    export LD_LIBRARY_PATH=/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cu13/lib:\${LD_LIBRARY_PATH}
    
    DATA_DIR='$DATA_DIR'
    if [ ! -d \"\$DATA_DIR\" ]; then
        DATA_DIR='/workspace/sparkpack/vla_teleop_data'
    fi
    
    /workspace/isaaclab/_isaac_sim/python.sh \
        /workspace/sparkpack/openarm_isaac_lab_trainer/scripts/reinforcement_learning/rsl_rl/play_bimanual_training_data.py \
        \"\$DATA_DIR\" \
        --replay \
        --real-time \
        ${EXTRA_ARGS}
"
