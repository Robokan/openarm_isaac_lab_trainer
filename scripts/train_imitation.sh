#!/bin/bash
#
# OpenArm Imitation Learning Training Script
# Uses LeRobot ACT policy for bimanual manipulation
#
# Usage:
#   ./train_imitation.sh                    # Train with defaults
#   ./train_imitation.sh --resume           # Resume from last checkpoint
#   ./train_imitation.sh --gpu 0            # Use specific GPU
#
# Prerequisites:
#   - LeRobot installed (uv sync in openpi directory)
#   - Training data in vla_teleop_data_lerobot/
#

set -e

# Configuration
GPU="${GPU:-1}"                    # Default to GPU 1 (GPU 0 often has Riva)
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASET_REPO="openarm-teleop"
DATASET_ROOT="/home/bizon/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data_lerobot"
OUTPUT_DIR="/home/bizon/sparkpack/openpi/outputs/openarm_act"
POLICY_TYPE="act"

# Parse arguments
RESUME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="--resume"
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--resume] [--gpu N] [--batch-size N]"
            exit 1
            ;;
    esac
done

# Header
echo "========================================"
echo "OpenArm Imitation Learning Training"
echo "========================================"
echo ""
echo "GPU: $GPU"
echo "Batch Size: $BATCH_SIZE"
echo "Dataset: $DATASET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Policy: $POLICY_TYPE"
if [ -n "$RESUME" ]; then
    echo "Mode: RESUME from checkpoint"
else
    echo "Mode: Fresh training"
fi
echo ""

# Check dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    echo "Please run data collection first."
    exit 1
fi

# Check GPU availability
if ! nvidia-smi -i $GPU &>/dev/null; then
    echo "ERROR: GPU $GPU not available"
    nvidia-smi --list-gpus
    exit 1
fi

# Show GPU info
echo "GPU $GPU:"
nvidia-smi -i $GPU --query-gpu=name,memory.free --format=csv,noheader
echo ""

# Activate environment
cd /home/bizon/sparkpack/openpi

echo "Starting training..."
echo "Press Ctrl+C to stop (checkpoints are saved periodically)"
echo ""

# Run training
CUDA_VISIBLE_DEVICES=$GPU uv run python -m lerobot.scripts.train \
    --dataset.repo_id=$DATASET_REPO \
    --dataset.root=$DATASET_ROOT \
    --policy.type=$POLICY_TYPE \
    --batch_size=$BATCH_SIZE \
    --output_dir=$OUTPUT_DIR \
    $RESUME

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
