#!/usr/bin/env bash
# Install persistent MATLAB WSL Host ID fix: systemd oneshot + passwordless sudo for login fallback.
# Run once per WSL machine: sudo bash scripts/matlab/install-wsl-hostid-persist.sh
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
