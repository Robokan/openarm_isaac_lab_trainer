#!/usr/bin/env bash
# Start WiVRn VR server (Flatpak)
#
# This server must be running before using XR teleoperation.
# Connect your VR headset (e.g., Vive XR Elite) to this server,
# then run: ./scripts/teleop_xr.sh

echo "Starting WiVRn VR server..."
echo "Connect your VR headset to this computer's IP address."
echo ""

flatpak run io.github.wivrn.wivrn
