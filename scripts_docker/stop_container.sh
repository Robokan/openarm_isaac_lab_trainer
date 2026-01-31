#!/usr/bin/env bash
# Stop and remove the Isaac Lab container

CONTAINER_NAME="isaac-lab"

echo "Stopping container '${CONTAINER_NAME}'..."
docker stop ${CONTAINER_NAME} 2>/dev/null && echo "Container stopped."
docker rm ${CONTAINER_NAME} 2>/dev/null && echo "Container removed."
echo "Done."
