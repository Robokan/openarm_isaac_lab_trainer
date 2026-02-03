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
#   INTERNET_CONN="Aloha house 1"  # Connection to bring up on USB adapter for internet

HOTSPOT_CONN="${HOTSPOT_CONN:-Hotspot}"
INTERNET_CONN="${INTERNET_CONN:-Aloha house 1}"
USB_WIFI_DEV="${USB_WIFI_DEV:-wlx54ef33ebd114}"

# Get the interface the hotspot is configured to use
HOTSPOT_DEV="$(nmcli -g connection.interface-name con show "${HOTSPOT_CONN}" 2>/dev/null || true)"
if [[ -z "${HOTSPOT_DEV}" || "${HOTSPOT_DEV}" == "--" ]]; then
  echo "[ERROR] Could not determine hotspot interface from connection '${HOTSPOT_CONN}'"
  exit 1
fi

echo "[INFO] Hotspot will use interface: ${HOTSPOT_DEV}"

# Find and disconnect any active connection on the hotspot interface
ACTIVE_CONN="$(nmcli -g GENERAL.CONNECTION device show "${HOTSPOT_DEV}" 2>/dev/null || true)"
if [[ -n "${ACTIVE_CONN}" && "${ACTIVE_CONN}" != "--" && "${ACTIVE_CONN}" != "${HOTSPOT_CONN}" ]]; then
  echo "[INFO] Disconnecting '${ACTIVE_CONN}' from ${HOTSPOT_DEV}"
  nmcli con down "${ACTIVE_CONN}" || true
fi

echo "[INFO] Bringing up hotspot: ${HOTSPOT_CONN}"
nmcli con up "${HOTSPOT_CONN}"

echo "[INFO] Disabling power save on ${HOTSPOT_DEV}"
sudo iw dev "${HOTSPOT_DEV}" set power_save off

# Reconnect internet on the USB Wi-Fi adapter
if nmcli device status | grep -q "^${USB_WIFI_DEV}"; then
  echo "[INFO] Connecting '${INTERNET_CONN}' on USB adapter ${USB_WIFI_DEV}"
  nmcli con up "${INTERNET_CONN}" ifname "${USB_WIFI_DEV}" 2>/dev/null || \
    echo "[WARN] Could not connect '${INTERNET_CONN}' on ${USB_WIFI_DEV}"
else
  echo "[WARN] USB Wi-Fi adapter ${USB_WIFI_DEV} not found, skipping internet connection"
fi

echo ""
echo "[OK] Hotspot is up on ${HOTSPOT_DEV}"
echo "     SSID: $(nmcli -s -g 802-11-wireless.ssid con show "${HOTSPOT_CONN}")"
USB_STATUS="$(nmcli -g GENERAL.STATE device show "${USB_WIFI_DEV}" 2>/dev/null || echo "not found")"
echo "     Internet via ${USB_WIFI_DEV}: ${USB_STATUS}"
