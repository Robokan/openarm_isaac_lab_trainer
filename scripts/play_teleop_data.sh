#!/bin/bash
# Play back recorded LeRobot teleoperation data
#
# Usage:
#   ./scripts/play_teleop_data.sh                    # Play all episodes
#   ./scripts/play_teleop_data.sh --episode 5        # Play specific episode
#   ./scripts/play_teleop_data.sh --episode 2-4      # Play range (2, 3, 4)
#   ./scripts/play_teleop_data.sh --episode 1,3,6    # Play list (1, 3, 6)
#   ./scripts/play_teleop_data.sh --loop             # Loop playback
#   ./scripts/play_teleop_data.sh --verify           # Just verify data, no simulation
#   ./scripts/play_teleop_data.sh --no-collect-video # Skip video capture during playback
#   ./scripts/play_teleop_data.sh --label            # Prompt for task label after each episode
#   ./scripts/play_teleop_data.sh --headless         # Run without GUI at max speed (no real-time limiting)
#
# Controls during playback:
#   Q     - Quit
#   N     - Next episode
#   SPACE - Pause/Resume

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Default data directory - check both possible locations
DATA_DIR="${PROJECT_DIR}/vla_teleop_data"
if [ ! -d "$DATA_DIR" ]; then
    # Check parent directory (old save location)
    DATA_DIR="$(dirname "$PROJECT_DIR")/vla_teleop_data"
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Record some teleoperation data first using teleop_bimanual.sh or teleop_xr.sh"
    exit 1
fi

# List available episodes
EPISODES_DIR="${DATA_DIR}/episodes"
if [ -d "$EPISODES_DIR" ]; then
    echo "Available episodes in ${DATA_DIR}:"
    for ep in "$EPISODES_DIR"/episode_*; do
        if [ -d "$ep" ]; then
            ep_name=$(basename "$ep")
            # Count frames if data.parquet exists
            if [ -f "$ep/data.parquet" ]; then
                echo "  - $ep_name"
            fi
        fi
    done
    echo ""
fi

# Source Isaac Lab environment
if [ -f "${PROJECT_DIR}/scripts/source_me.sh" ]; then
    source "${PROJECT_DIR}/scripts/source_me.sh"
else
    echo "Error: Cannot find scripts/source_me.sh"
    echo "Make sure Isaac Lab is set up correctly."
    exit 1
fi

# Run the playback script
# Note: --headless automatically disables real-time mode for max speed
python "${PROJECT_DIR}/scripts/reinforcement_learning/rsl_rl/play_bimanual_training_data.py" \
    "$DATA_DIR" \
    --replay \
    --real-time \
    "$@"
