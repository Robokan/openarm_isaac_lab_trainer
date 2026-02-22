#!/bin/bash
# Switch between Riva versions
# Usage: ./riva_switch.sh [fast|quality]
#   fast    = Riva 2.18 with FastPitch (faster, less natural)
#   quality = Riva 2.19 with Magpie (slower, more natural)

RIVA_BASE=~/riva

show_usage() {
    echo "Usage: $0 [fast|quality|status]"
    echo ""
    echo "  fast    - Use Riva 2.18 with FastPitch TTS (faster response)"
    echo "  quality - Use Riva 2.19 with Magpie TTS (more natural voice)"
    echo "  status  - Show which Riva version is currently running"
    echo ""
}

get_status() {
    local running=$(docker ps --format '{{.Image}}' | grep riva-speech | head -1)
    if [[ -z "$running" ]]; then
        echo "No Riva server running"
        return 1
    elif [[ "$running" == *"2.18"* ]]; then
        echo "Riva 2.18 (FastPitch - fast)"
    elif [[ "$running" == *"2.19"* ]]; then
        echo "Riva 2.19 (Magpie - quality)"
    else
        echo "Unknown Riva version: $running"
    fi
}

stop_riva() {
    echo "Stopping current Riva server..."
    docker stop riva-speech 2>/dev/null
    docker rm riva-speech 2>/dev/null
    sleep 2
}

start_fast() {
    echo "Starting Riva 2.18 (FastPitch - fast)..."
    cd "$RIVA_BASE/riva_quickstart_v2.18.0"
    bash riva_start.sh
    echo ""
    echo "Riva 2.18 started. Use voice_name='English-US.Male-1' in your code."
}

start_quality() {
    echo "Starting Riva 2.19 (Magpie - quality)..."
    cd "$RIVA_BASE/riva_quickstart_v2.19.0"
    bash riva_start.sh
    echo ""
    echo "Riva 2.19 started. Use voice_name='Magpie-Multilingual.EN-US.Male.Male-1' in your code."
}

case "$1" in
    fast)
        stop_riva
        start_fast
        ;;
    quality)
        stop_riva
        start_quality
        ;;
    status)
        get_status
        ;;
    *)
        show_usage
        echo "Current status:"
        get_status
        ;;
esac
