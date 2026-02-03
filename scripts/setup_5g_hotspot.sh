#!/usr/bin/env bash
set -euo pipefail

# Setup dedicated 5GHz hotspot for XR streaming.
# Assumes a NetworkManager connection named "Hotspot" already exists.
#
# Usage:
#   ./scripts/setup_5g_hotspot.sh
#
# Optional env overrides:
#   HOTSPOT_CONN="Hotspot"
#   CLIENT_CONN="Aloha house 1"   # Wi-Fi connection to disconnect (optional)

HOTSPOT_CONN="${HOTSPOT_CONN:-Hotspot}"
CLIENT_CONN="${CLIENT_CONN:-Aloha house 1}"

echo "[INFO] Bringing down client Wi-Fi (if active): ${CLIENT_CONN}"
nmcli -t -f NAME,ACTIVE con show | grep -q "^${CLIENT_CONN}:yes$" && nmcli con down "${CLIENT_CONN}" || true

echo "[INFO] Bringing up hotspot: ${HOTSPOT_CONN}"
nmcli con up "${HOTSPOT_CONN}"

HOTSPOT_DEV="$(nmcli -g connection.interface-name con show "${HOTSPOT_CONN}" || true)"
if [[ -z "${HOTSPOT_DEV}" || "${HOTSPOT_DEV}" == "--" ]]; then
  echo "[WARN] Could not determine hotspot device. Run 'nmcli device status' to verify."
  exit 0
fi

echo "[INFO] Disabling power save on ${HOTSPOT_DEV}"
sudo iw dev "${HOTSPOT_DEV}" set power_save off

echo ""
echo "[OK] Hotspot is up on ${HOTSPOT_DEV}"
echo "     SSID: $(nmcli -s -g 802-11-wireless.ssid con show "${HOTSPOT_CONN}")"
