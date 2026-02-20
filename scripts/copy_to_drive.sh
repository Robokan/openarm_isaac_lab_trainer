#!/bin/bash
# Copy OpenPi training files to external drive for DGX Spark
#
# Usage:
#   ./scripts/copy_to_drive.sh /media/bizon/MY_DRIVE
#   ./scripts/copy_to_drive.sh /mnt/external

set -e

DRIVE_PATH="${1:-}"

if [ -z "$DRIVE_PATH" ]; then
    echo "Usage: $0 <drive_path>"
    echo "Example: $0 /media/bizon/MY_EXTERNAL_DRIVE"
    exit 1
fi

if [ ! -d "$DRIVE_PATH" ]; then
    echo "Error: Drive path does not exist: $DRIVE_PATH"
    exit 1
fi

DEST="$DRIVE_PATH/openpi_training"
echo "========================================"
echo "Copying OpenPi training files to drive"
echo "========================================"
echo "Destination: $DEST"
echo ""

mkdir -p "$DEST"

# 1. OpenPi repository (excluding venv, cache, wandb logs)
echo "[1/5] Copying OpenPi repo..."
rsync -av --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='wandb' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    /home/bizon/sparkpack/openpi/ "$DEST/openpi/"

# 2. LeRobot dataset
echo ""
echo "[2/5] Copying LeRobot dataset..."
rsync -av --progress \
    /home/bizon/sparkpack/openarm_isaac_lab_trainer/vla_teleop_data_lerobot/ \
    "$DEST/vla_teleop_data_lerobot/"

# 3. Cached base model checkpoints (Pi 0 and Pi 0.5)
echo ""
echo "[3/6] Copying base model checkpoints..."
mkdir -p "$DEST/cache/checkpoints"

# Copy Pi 0.5 base
if [ -d "/home/bizon/.cache/openpi/openpi-assets/checkpoints/pi05_base" ]; then
    echo "  Copying pi05_base..."
    rsync -av --progress \
        /home/bizon/.cache/openpi/openpi-assets/checkpoints/pi05_base/ \
        "$DEST/cache/checkpoints/pi05_base/"
else
    echo "  pi05_base not found (will download on first run)"
fi

# Copy Pi 0 base
if [ -d "/home/bizon/.cache/openpi/openpi-assets/checkpoints/pi0_base" ]; then
    echo "  Copying pi0_base..."
    rsync -av --progress \
        /home/bizon/.cache/openpi/openpi-assets/checkpoints/pi0_base/ \
        "$DEST/cache/checkpoints/pi0_base/"
else
    echo "  pi0_base not found (will download on first run)"
fi

# 4. Norm stats
echo ""
echo "[4/6] Copying norm stats..."
if [ -d "/home/bizon/sparkpack/openpi/assets/pi05_openarm" ]; then
    mkdir -p "$DEST/assets"
    rsync -av --progress \
        /home/bizon/sparkpack/openpi/assets/pi05_openarm/ \
        "$DEST/assets/pi05_openarm/"
else
    echo "  Warning: Norm stats not found. Run compute_norm_stats.py first."
fi

# 5. Trained checkpoints (if any exist)
echo ""
echo "[5/6] Copying trained checkpoints (if any)..."
if [ -d "/home/bizon/sparkpack/openpi/checkpoints/pi05_openarm" ]; then
    mkdir -p "$DEST/trained_checkpoints"
    rsync -av --progress \
        /home/bizon/sparkpack/openpi/checkpoints/pi05_openarm/ \
        "$DEST/trained_checkpoints/pi05_openarm/"
else
    echo "  No trained checkpoints found (training may still be in progress)."
fi

# 6. Copy the copy_from_drive.sh script for use on Spark
echo ""
echo "[6/6] Copying setup script for Spark..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/copy_from_drive.sh" "$DEST/copy_from_drive.sh"
chmod +x "$DEST/copy_from_drive.sh"
echo "  Copied copy_from_drive.sh to drive"

echo ""
echo "========================================"
echo "Copy complete!"
echo "========================================"
echo ""
echo "Total size:"
du -sh "$DEST"
echo ""
echo "Contents:"
du -sh "$DEST"/*
echo ""
echo "Next steps on DGX Spark:"
echo "  1. Mount the drive"
echo "  2. cd /path/to/drive/openpi_training"
echo "  3. ./copy_from_drive.sh /path/to/drive /home/nvidia"
