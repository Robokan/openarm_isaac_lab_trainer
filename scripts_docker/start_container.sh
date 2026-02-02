#!/usr/bin/env bash
set -e

# OpenArm Isaac Lab Container Launcher
# Starts the Isaac Lab container with X11 forwarding for GUI support

IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"
CONTAINER_NAME="isaac-lab"
DISPLAY_NUM="${DISPLAY:-:1}"

# Get repo root (parent of scripts_docker)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' already exists."
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is running. Use 'docker exec -it ${CONTAINER_NAME} bash' to attach."
        exit 0
    else
        echo "Starting existing container..."
        docker start ${CONTAINER_NAME}
        exit 0
    fi
fi

# Create cache directories if they don't exist
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
mkdir -p ~/docker/isaac-sim/{logs,data,documents}

# Enable X11 forwarding
xhost +local:docker >/dev/null 2>&1

echo "Starting Isaac Lab container with X11 forwarding..."

docker run --name ${CONTAINER_NAME} -d --gpus all \
   -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY="${DISPLAY_NUM}" \
   -e NVIDIA_DRIVER_CAPABILITIES=all \
   -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
   -e XR_RUNTIME_JSON=/root/.config/openxr/1/active_runtime.json \
   --privileged \
   -v /dev/input:/dev/input \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v /run/user/$(id -u):/run/user/$(id -u) \
   -v /tmp/.steam.pipe:/tmp/.steam.pipe \
   -v $HOME/.steam:/root/.steam:ro \
   -v $HOME/.config/openxr:/root/.config/openxr:ro \
   -v /var/lib/flatpak:/var/lib/flatpak:ro \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v "${REPO_ROOT}":/workspace/openarm_isaac_lab:rw \
   --entrypoint bash \
   ${IMAGE} \
   -c "tail -f /dev/null"

echo "Container started. Installing OpenArm package..."

# Install OpenArm package
docker exec ${CONTAINER_NAME} bash -c "cd /workspace/openarm_isaac_lab && /workspace/isaaclab/isaaclab.sh -p -m pip install -e source/openarm" 2>&1 | tail -5

echo ""
echo "=========================================="
echo "Container '${CONTAINER_NAME}' is ready!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  ./train_reach.sh          # Train unimanual reach"
echo "  ./train_lift.sh           # Train lift cube"
echo "  ./train_drawer.sh         # Train open drawer"
echo "  ./train_bimanual_reach.sh # Train bimanual reach"
echo ""
echo "  ./play_reach.sh           # Play trained reach model"
echo "  ./play_lift.sh            # Play trained lift model"
echo "  ./play_drawer.sh          # Play trained drawer model"
echo "  ./play_bimanual_reach.sh  # Play trained bimanual model"
echo ""
echo "  ./stop_container.sh       # Stop the container"
echo "  docker exec -it ${CONTAINER_NAME} bash  # Interactive shell"
