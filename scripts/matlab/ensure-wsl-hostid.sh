#!/usr/bin/env bash
# WSL2 gives eth0 a new MAC on many restarts; node-locked MATLAB licenses bind to a
# fixed Host ID. Create bond0 with the MAC from the active license so batch mode works.
# See docs/matlab.md §8 (Licensing Error 9) and matlab-license.md on this machine.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ] || ! grep -qi microsoft /proc/version 2>/dev/null; then
  exit 0
fi

_mac="${MATLAB_WSL_BOND_MAC:-00:15:5d:49:c1:7b}"
_log="${MATLAB_WSL_HOSTID_LOG:-/var/log/matlab-wsl-hostid.log}"

_bond_mac() {
  ip link show bond0 2>/dev/null | awk '/link\/ether/ {print $2; exit}' || true
}

_log_line() {
  if [ "$(id -u)" -eq 0 ]; then
    echo "$(date -Is) $*" >>"$_log" 2>/dev/null || true
  fi
}

_check_only=0
for _arg in "$@"; do
  case "$_arg" in
    --check) _check_only=1 ;;
  esac
done

_current="$(_bond_mac)"
if [ "$_current" = "$_mac" ]; then
  exit 0
fi

if [ "$_check_only" -eq 1 ]; then
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "ensure-wsl-hostid.sh: bond0 MAC is ${_current:-<missing>}, want $_mac — re-run with sudo" >&2
  exit 1
fi

if ip link show bond0 >/dev/null 2>&1; then
  ip link set dev bond0 down 2>/dev/null || true
  if ! ip link set dev bond0 address "$_mac" 2>/dev/null; then
    ip link del bond0 2>/dev/null || true
  fi
fi

if ! ip link show bond0 >/dev/null 2>&1; then
  ip link add bond0 address "$_mac" type bond
fi

_current="$(_bond_mac)"
if [ "$_current" != "$_mac" ]; then
  _log_line "fail want=$_mac got=${_current:-<missing>}"
  echo "ensure-wsl-hostid.sh: bond0 MAC is still ${_current:-<missing>}, want $_mac" >&2
  exit 1
fi

_log_line "ok bond0=$_mac"
