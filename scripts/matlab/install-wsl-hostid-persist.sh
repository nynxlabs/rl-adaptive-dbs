#!/usr/bin/env bash
# Install persistent MATLAB WSL Host ID fix: systemd oneshot + passwordless sudo for login fallback.
# Run once per WSL machine: sudo bash scripts/matlab/install-wsl-hostid-persist.sh [--mac XX:XX:XX:XX:XX:XX]
# Without --mac, the licensed MAC is read from the newest node-locked MATLAB license
# file of the invoking user (MATLAB_HOSTID=<mac> in *.lic).
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo bash $0" >&2
  exit 1
fi

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ensure="$_script_dir/ensure-wsl-hostid.sh"
_unit_src="$_script_dir/matlab-wsl-hostid.service"
_unit_dst="/etc/systemd/system/matlab-wsl-hostid.service"
_sudoers="/etc/sudoers.d/matlab-wsl-hostid"
_repo_root="$(cd "$_script_dir/../.." && pwd)"
_user="${SUDO_USER:-}"
if [ -z "$_user" ] || [ "$_user" = root ]; then
  echo "Run via sudo from your normal account (SUDO_USER is unset)" >&2
  exit 1
fi

_mac=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mac) _mac="${2:-}"; shift 2 ;;
    --mac=*) _mac="${1#--mac=}"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$_mac" ]; then
  _home="$(getent passwd "$_user" | cut -d: -f6)"
  _lic="$(ls -t "$_home"/.matlab/R*_licenses/*.lic "${MATLAB_ROOT:-$_home/MATLAB}"/licenses/*.lic 2>/dev/null \
    | while read -r _f; do grep -qE 'MATLAB_HOSTID=[0-9A-Fa-f]{12}' "$_f" && { echo "$_f"; break; }; done || true)"
  if [ -z "$_lic" ]; then
    echo "No node-locked MATLAB license with a MATLAB_HOSTID found for $_user; pass --mac" >&2
    exit 1
  fi
  _hex="$(grep -oE 'MATLAB_HOSTID=[0-9A-Fa-f]{12}' "$_lic" | head -1 | cut -d= -f2)"
  _mac="$(printf '%s' "$_hex" | sed -E 's/(..)(..)(..)(..)(..)(..)/\1:\2:\3:\4:\5:\6/')"
  echo "Licensed MAC $_mac from $_lic"
fi
_mac="$(printf '%s' "$_mac" | tr '[:upper:]' '[:lower:]')"
if ! printf '%s' "$_mac" | grep -qE '^([0-9a-f]{2}:){5}[0-9a-f]{2}$'; then
  echo "Invalid MAC: $_mac" >&2
  exit 1
fi

_config="/etc/default/matlab-wsl-hostid"
printf 'MATLAB_WSL_BOND_MAC=%s\n' "$_mac" > "$_config"
chmod 0644 "$_config"

chmod +x "$_ensure"

sed "s|@REPO_ROOT@|$_repo_root|g" "$_unit_src" > "$_unit_dst"
chmod 0644 "$_unit_dst"
printf '%s\n' "$_user ALL=(root) NOPASSWD: $_ensure" > "$_sudoers"
chmod 0440 "$_sudoers"
visudo -c -f "$_sudoers"

systemctl daemon-reload
systemctl enable matlab-wsl-hostid.service
systemctl start matlab-wsl-hostid.service

echo "=== matlab-wsl-hostid installed ==="
systemctl status matlab-wsl-hostid.service --no-pager || true
ip link show bond0 2>/dev/null | grep link/ether || true
echo "Log: /var/log/matlab-wsl-hostid.log"
echo "Optional: remove duplicate [boot] command= from /etc/wsl.conf (systemd handles this)."
