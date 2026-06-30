#!/usr/bin/env bash
# Install persistent MATLAB WSL Host ID fix: systemd oneshot + passwordless sudo for login fallback.
# Run once on nynxbox (WSL): sudo bash scripts/matlab/install-wsl-hostid-persist.sh
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

chmod +x "$_ensure"

install -m 0644 "$_unit_src" "$_unit_dst"
printf '%s\n' "nynxbox ALL=(root) NOPASSWD: $_ensure" > "$_sudoers"
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
