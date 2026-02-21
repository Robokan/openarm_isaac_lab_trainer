#!/usr/bin/env bash
# NVIDIA Riva ASR Setup Script
#
# One-time setup for NVIDIA Riva speech recognition.
# This downloads the Riva container and models (~25GB total).
#
# Prerequisites:
#   - NVIDIA GPU with 8GB+ VRAM
#   - Docker with NVIDIA runtime
#   - NGC CLI (optional, for faster downloads)
#
# Usage:
#   ./riva_setup.sh [install_dir]
#   ./riva_setup.sh /home/bizon/riva

set -e

INSTALL_DIR="${1:-$HOME/riva}"
RIVA_VERSION="2.18.0"

echo "========================================"
echo "NVIDIA Riva ASR Setup"
echo "========================================"
echo "Install directory: $INSTALL_DIR"
echo "Riva version: $RIVA_VERSION"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found. Please install Docker first."
    echo "  sudo apt-get install docker.io"
    echo "  sudo usermod -aG docker $USER"
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "[WARN] NVIDIA Docker runtime may not be configured."
    echo "  Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download Riva quickstart if not present
if [ ! -d "riva_quickstart_v${RIVA_VERSION}" ] && [ ! -d "riva_quickstart_arm64_v${RIVA_VERSION}" ]; then
    echo ""
    echo "[1/4] Downloading Riva quickstart..."
    
    # Try NGC CLI first
    if command -v ngc &> /dev/null; then
        echo "Using NGC CLI..."
        ngc registry resource download-version "nvidia/riva/riva_quickstart:${RIVA_VERSION}"
    else
        echo ""
        echo "=============================================="
        echo "NGC CLI not installed - Manual download required"
        echo "=============================================="
        echo ""
        echo "Option 1: Install NGC CLI (recommended)"
        echo "  wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip"
        echo "  unzip ngccli_linux.zip"
        echo "  chmod +x ngc-cli/ngc"
        echo "  sudo mv ngc-cli/ngc /usr/local/bin/"
        echo "  ngc config set"
        echo ""
        echo "  Then run this script again."
        echo ""
        echo "Option 2: Download manually from NVIDIA"
        echo "  1. Go to: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart"
        echo "  2. Sign in with NVIDIA account (free)"
        echo "  3. Download version ${RIVA_VERSION}"
        echo "  4. Extract to: $INSTALL_DIR"
        echo "  5. Run this script again"
        echo ""
        echo "Option 3: Use Docker directly (skip quickstart)"
        echo "  docker pull nvcr.io/nvidia/riva/riva-speech:${RIVA_VERSION}"
        echo ""
        
        read -p "Install NGC CLI now? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Installing NGC CLI..."
            cd /tmp
            wget -q -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
            unzip -o ngccli_linux.zip
            chmod +x ngc-cli/ngc
            sudo mv ngc-cli/ngc /usr/local/bin/
            rm -rf ngccli_linux.zip ngc-cli
            
            echo ""
            echo "NGC CLI installed. Now configure it with your API key:"
            echo ""
            echo "  1. Go to: https://ngc.nvidia.com/setup/api-key"
            echo "  2. Generate an API key"
            echo "  3. Run: ngc config set"
            echo ""
            ngc config set
            
            # Try download again
            cd "$INSTALL_DIR"
            ngc registry resource download-version "nvidia/riva/riva_quickstart:${RIVA_VERSION}"
        else
            echo ""
            echo "Please download Riva manually and run this script again."
            exit 1
        fi
    fi
else
    echo "[1/4] Riva quickstart already downloaded."
fi

cd "riva_quickstart_v${RIVA_VERSION}" 2>/dev/null || cd riva_quickstart_arm64_v${RIVA_VERSION} 2>/dev/null || cd riva_quickstart*

# Configure for ASR only (saves VRAM and download time)
echo ""
echo "[2/4] Configuring Riva for ASR only..."

if [ -f "config.sh" ]; then
    # Backup original config
    cp config.sh config.sh.bak
    
    # Enable ASR and TTS (for voice responses)
    sed -i 's/service_enabled_asr=.*/service_enabled_asr=true/' config.sh
    sed -i 's/service_enabled_nlp=.*/service_enabled_nlp=false/' config.sh
    sed -i 's/service_enabled_tts=.*/service_enabled_tts=true/' config.sh
    sed -i 's/service_enabled_nmt=.*/service_enabled_nmt=false/' config.sh
    
    # Use English models
    sed -i 's/language_code=.*/language_code=("en-US")/' config.sh
    
    echo "  ASR and TTS enabled"
else
    echo "[WARN] config.sh not found, using defaults"
fi

# Initialize models
echo ""
echo "[3/4] Initializing Riva models (this may take 10-30 minutes)..."
echo "  Downloading ASR models from NGC..."
bash riva_init.sh

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "========================================"
echo "Riva ASR is ready!"
echo "========================================"
echo ""
echo "To start Riva server:"
echo "  cd $INSTALL_DIR/riva_quickstart_v${RIVA_VERSION}"
echo "  CUDA_VISIBLE_DEVICES=0 bash riva_start.sh"
echo ""
echo "To stop Riva server:"
echo "  bash riva_stop.sh"
echo ""
echo "Riva will listen on localhost:50051 (gRPC)"
