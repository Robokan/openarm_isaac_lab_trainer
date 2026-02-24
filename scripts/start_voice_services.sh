#!/bin/bash
# Start all voice assistant services (Riva + NIM)
# Then run: ./voice_assistant.sh  (or ./voice_web.sh for web interface)
#
# Usage:
#   ./start_voice_services.sh        # Start Riva and NIM
#   ./start_voice_services.sh --all  # Start everything including voice assistant

exec "$(dirname "$0")/voice_assistant/start_all.sh" "$@"
