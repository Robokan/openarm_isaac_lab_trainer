#!/usr/bin/env bash
# Activate the Isaac Lab environment with Isaac Sim paths
#
# Usage: source ./scripts/activate_env.sh

ISAAC_LAB_PATH="${ISAAC_LAB_PATH:-$HOME/sparkpack/IsaacLab}"

# Step 1: Activate the virtual environment
if [[ -f "${ISAAC_LAB_PATH}/env_isaaclab/bin/activate" ]]; then
    source "${ISAAC_LAB_PATH}/env_isaaclab/bin/activate"
    echo "Activated Isaac Lab environment from ${ISAAC_LAB_PATH}/env_isaaclab"
elif [[ -f "${ISAAC_LAB_PATH}/.venv/bin/activate" ]]; then
    source "${ISAAC_LAB_PATH}/.venv/bin/activate"
    echo "Activated Isaac Lab environment from ${ISAAC_LAB_PATH}/.venv"
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "Activated local .venv environment"
else
    echo "No virtual environment found."
    echo "Set ISAAC_LAB_PATH or create a .venv in this directory."
    return 1 2>/dev/null || exit 1
fi

# Step 2: Set up Isaac Sim paths (PYTHONPATH, CARB_APP_PATH, etc.)
if [[ -f "${ISAAC_LAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
    source "${ISAAC_LAB_PATH}/_isaac_sim/setup_conda_env.sh"
    echo "Isaac Sim paths configured"
fi
