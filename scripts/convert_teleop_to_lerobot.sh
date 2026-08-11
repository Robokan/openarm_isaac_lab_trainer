#!/usr/bin/env bash
# Convert VR teleop fallback data to native LeRobot v3.0
#
# Usage:
#   ./scripts/convert_teleop_to_lerobot.sh
#   ./scripts/convert_teleop_to_lerobot.sh --input vla_teleop_data --output vla_teleop_data_lerobot
#   ./scripts/convert_teleop_to_lerobot.sh --max-episodes 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LEROBOT_SRC="${REPO_ROOT}/../lerobot/src"

cd "$REPO_ROOT"

if [[ -d "$LEROBOT_SRC" ]]; then
    export PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Prefer the sibling lerobot venv (has LeRobot v3 + deps)
PYTHON="${PYTHON:-python3}"
if [[ -x "${REPO_ROOT}/../lerobot/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/../lerobot/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/../openpi/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/../openpi/.venv/bin/python"
fi

echo "Using Python: $PYTHON"
exec "$PYTHON" "${SCRIPT_DIR}/convert_teleop_to_lerobot.py" "$@"
