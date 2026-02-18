#!/usr/bin/env bash
# Play back recorded LeRobot teleoperation data (Docker version)
#
# Usage:
#   ./scripts_docker/play_teleop_data.sh                    # Play all episodes
#   ./scripts_docker/play_teleop_data.sh --episode 0        # Play specific episode
#   ./scripts_docker/play_teleop_data.sh --loop             # Loop playback
#   ./scripts_docker/play_teleop_data.sh --verify           # Just verify data, no simulation
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

# Data directory inside container
DATA_DIR="/workspace/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data"

# Check if data exists
docker exec ${CONTAINER_NAME} bash -c "
    if [ ! -d '$DATA_DIR' ]; then
        # Check alternate location
        ALT_DIR='/workspace/sparkpack/vla_teleop_data'
        if [ -d \"\$ALT_DIR\" ]; then
            DATA_DIR=\"\$ALT_DIR\"
        else
            echo 'Error: No data directory found at:'
            echo '  $DATA_DIR'
            echo '  \$ALT_DIR'
            echo 'Record some teleoperation data first.'
            exit 1
        fi
    fi
    
    # List available episodes
    EPISODES_DIR=\"\${DATA_DIR:-$DATA_DIR}/episodes\"
    if [ -d \"\$EPISODES_DIR\" ]; then
        echo 'Available episodes:'
        for ep in \"\$EPISODES_DIR\"/episode_*; do
            if [ -d \"\$ep\" ]; then
                echo \"  - \$(basename \$ep)\"
            fi
        done
        echo ''
    fi
"

# Run the playback script
docker exec -it ${CONTAINER_NAME} bash -c "
    cd /workspace/isaaclab && source /workspace/isaaclab/_isaac_sim/setup_conda_env.sh
    
    DATA_DIR='$DATA_DIR'
    if [ ! -d \"\$DATA_DIR\" ]; then
        DATA_DIR='/workspace/sparkpack/vla_teleop_data'
    fi
    
    /workspace/isaaclab/_isaac_sim/python.sh \
        /workspace/sparkpack/openarm_isaac_lab_trainer/scripts/reinforcement_learning/rsl_rl/play_bimanual_training_data.py \
        \"\$DATA_DIR\" \
        --replay \
        --real-time \
        $@
"
