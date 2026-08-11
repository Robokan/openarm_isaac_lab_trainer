#!/usr/bin/env bash
# Activate the Isaac Lab environment with Isaac Sim paths
#
# Usage: source ./scripts/source_me.sh

ISAAC_LAB_PATH="${ISAAC_LAB_PATH:-$HOME/sparkpack/IsaacLab}"

# Step 1: Activate the Isaac Lab virtual environment (not this repo's uv .venv)
if [[ -f "${ISAAC_LAB_PATH}/env_isaaclab/bin/activate" ]]; then
    source "${ISAAC_LAB_PATH}/env_isaaclab/bin/activate"
    echo "Activated Isaac Lab environment from ${ISAAC_LAB_PATH}/env_isaaclab"
elif [[ -f "${ISAAC_LAB_PATH}/.venv/bin/activate" ]]; then
    source "${ISAAC_LAB_PATH}/.venv/bin/activate"
    echo "Activated Isaac Lab environment from ${ISAAC_LAB_PATH}/.venv"
else
    echo "No Isaac Lab Python environment found at:"
    echo "  ${ISAAC_LAB_PATH}/env_isaaclab"
    echo "  ${ISAAC_LAB_PATH}/.venv"
    if [[ -L "${ISAAC_LAB_PATH}/_isaac_sim" && ! -e "${ISAAC_LAB_PATH}/_isaac_sim" ]]; then
        echo ""
        echo "Note: ${ISAAC_LAB_PATH}/_isaac_sim is a broken symlink (container-only path)."
        echo "On this machine, run teleop via Docker instead:"
        echo "  ./scripts_docker/start_container.sh"
        echo "  ./scripts_docker/teleop_bimanual.sh --input keyboard"
    fi
    return 1 2>/dev/null || exit 1
fi

# Step 2: Set up Isaac Sim paths (PYTHONPATH, CARB_APP_PATH, etc.)
if [[ -f "${ISAAC_LAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
    source "${ISAAC_LAB_PATH}/_isaac_sim/setup_conda_env.sh"
    echo "Isaac Sim paths configured"
elif [[ -L "${ISAAC_LAB_PATH}/_isaac_sim" && ! -e "${ISAAC_LAB_PATH}/_isaac_sim" ]]; then
    echo "Error: ${ISAAC_LAB_PATH}/_isaac_sim is a broken symlink to $(readlink "${ISAAC_LAB_PATH}/_isaac_sim")."
    echo "Isaac Sim is only available inside Docker on this machine."
    echo "Use: ./scripts_docker/teleop_bimanual.sh --input keyboard"
    return 1 2>/dev/null || exit 1
fi
