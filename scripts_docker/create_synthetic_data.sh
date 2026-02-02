#!/usr/bin/env bash
set -e

# Synthetic Data Generation for Bimanual OpenArm (Docker)
# Drops cubes randomly on the table for data collection
#
# Usage:
#   ./create_synthetic_data.sh                    # Run with defaults
#   ./create_synthetic_data.sh --num_episodes 100 # More episodes
#   ./create_synthetic_data.sh --headless         # No GUI

CONTAINER_NAME="isaac-lab"
TASK="Isaac-Reach-OpenArm-Bi-v0"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please run ./start_container.sh first."
    exit 1
fi

# Check if checkpoint is specified in args
CHECKPOINT_ARG=""
HAS_CHECKPOINT=false
for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        HAS_CHECKPOINT=true
        break
    fi
done

# Find latest checkpoint if not specified
if [[ "$HAS_CHECKPOINT" == "false" ]]; then
    LATEST=$(docker exec ${CONTAINER_NAME} bash -c "find /workspace/openarm_isaac_lab/logs/rsl_rl -name 'model_*.pt' 2>/dev/null | grep -E '(bimanual|bi_reach|openarm_bi)' | sort -V | tail -1")
    
    if [[ -n "$LATEST" ]]; then
        echo "Using latest bimanual checkpoint: $LATEST"
        CHECKPOINT_ARG="--checkpoint $LATEST"
    else
        echo "=========================================="
        echo "WARNING: No trained bimanual model found!"
        echo "=========================================="
        echo ""
        echo "Please train a bimanual reach model first:"
        echo "  ./train_bimanual_reach.sh --headless"
        echo ""
        echo "Or specify a checkpoint manually:"
        echo "  ./create_synthetic_data.sh --checkpoint /path/to/model.pt"
        echo ""
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "SYNTHETIC DATA GENERATION - BIMANUAL"
echo "=========================================="
echo ""
echo "This script:"
echo "  1. Drops cubes randomly on the table"
echo "  2. Runs the bimanual reach policy"
echo "  3. (Future) Records data for VLA training"
echo ""
echo "Args: $@"
echo ""

docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p ./scripts/teleoperation/create_synthetic_data_bimanual.py --task ${TASK} ${CHECKPOINT_ARG} $*"
