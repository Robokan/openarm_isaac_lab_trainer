#!/bin/bash
# Copy OpenPi training files from external drive to DGX Spark
#
# Run this script ON THE DGX SPARK after connecting the external drive.
#
# Usage:
#   ./copy_from_drive.sh /media/nvidia/MY_DRIVE
#   ./copy_from_drive.sh /media/nvidia/MY_DRIVE /home/nvidia

set -e

DRIVE_PATH="${1:-}"
SPARK_HOME="${2:-$HOME}"

if [ -z "$DRIVE_PATH" ]; then
    echo "Usage: $0 <drive_path> [spark_home]"
    echo "Example: $0 /media/nvidia/MY_EXTERNAL_DRIVE"
    echo "Example: $0 /media/nvidia/MY_EXTERNAL_DRIVE /home/nvidia"
    exit 1
fi

SOURCE="$DRIVE_PATH/openpi_training"

if [ ! -d "$SOURCE" ]; then
    echo "Error: Source path does not exist: $SOURCE"
    echo "Make sure the drive is mounted and contains openpi_training folder."
    exit 1
fi

echo "========================================"
echo "Copying OpenPi training files to Spark"
echo "========================================"
echo "Source: $SOURCE"
echo "Destination: $SPARK_HOME"
echo ""

# 1. OpenPi repository
echo "[1/5] Copying OpenPi repo..."
rsync -av --progress "$SOURCE/openpi/" "$SPARK_HOME/openpi/"

# 2. LeRobot dataset
echo ""
echo "[2/5] Copying LeRobot dataset..."
mkdir -p "$SPARK_HOME/datasets"
rsync -av --progress "$SOURCE/vla_teleop_data_lerobot/" \
    "$SPARK_HOME/datasets/vla_teleop_data_lerobot/"

# 3. Base model checkpoint to cache
echo ""
echo "[3/5] Copying Pi 0.5 base model checkpoint to cache..."
if [ -d "$SOURCE/cache/checkpoints/pi05_base" ]; then
    mkdir -p "$SPARK_HOME/.cache/openpi/openpi-assets/checkpoints"
    rsync -av --progress "$SOURCE/cache/checkpoints/pi05_base/" \
        "$SPARK_HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base/"
else
    echo "  Base checkpoint not on drive - will download on first run."
fi

# 4. Norm stats
echo ""
echo "[4/5] Copying norm stats..."
if [ -d "$SOURCE/assets/pi05_openarm" ]; then
    mkdir -p "$SPARK_HOME/openpi/assets"
    rsync -av --progress "$SOURCE/assets/pi05_openarm/" \
        "$SPARK_HOME/openpi/assets/pi05_openarm/"
else
    echo "  Norm stats not on drive - need to compute on Spark."
fi

# 5. Trained checkpoints (if any)
echo ""
echo "[5/5] Copying trained checkpoints (if any)..."
if [ -d "$SOURCE/trained_checkpoints" ]; then
    mkdir -p "$SPARK_HOME/openpi/checkpoints"
    rsync -av --progress "$SOURCE/trained_checkpoints/" \
        "$SPARK_HOME/openpi/checkpoints/"
else
    echo "  No trained checkpoints on drive."
fi

# Update config.py with new dataset path
DATASET_PATH="$SPARK_HOME/datasets/vla_teleop_data_lerobot"
CONFIG_FILE="$SPARK_HOME/openpi/src/openpi/training/config.py"

echo ""
echo "========================================"
echo "Copy complete!"
echo "========================================"
echo ""
echo "IMPORTANT: Update the dataset path in config.py"
echo ""
echo "Edit: $CONFIG_FILE"
echo "Change local_dir to: $DATASET_PATH"
echo ""
echo "Or run this sed command:"
echo "  sed -i 's|/home/bizon/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data_lerobot|$DATASET_PATH|g' $CONFIG_FILE"
echo ""
echo "Then to train:"
echo "  cd $SPARK_HOME/openpi"
echo "  uv sync                    # Install dependencies"
echo "  uv run scripts/train.py pi05_openarm --exp-name spark_v1"
echo ""
echo "DGX Spark has unified memory, so you may not need FSDP:"
echo "  uv run scripts/train.py pi05_openarm --exp-name spark_v1 --fsdp-devices 1"
