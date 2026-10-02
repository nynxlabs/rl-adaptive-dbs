#!/usr/bin/env bash
# Repair wedged Multipass on Windows (elevated). See scripts/validation/repair-multipass.ps1
set -euo pipefail
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/../.." && pwd)"
distro="${WSL_DISTRO_NAME:-Ubuntu}"
# Windows %LOCALAPPDATA% as a WSL path (override with RL_DBS_WIN_LOCALAPPDATA).
win_localappdata() {
  if [[ -n "${RL_DBS_WIN_LOCALAPPDATA:-}" ]]; then
    printf '%s\n' "$RL_DBS_WIN_LOCALAPPDATA"
    return
  fi
  local win
  win="$(/mnt/c/Windows/System32/cmd.exe /c 'echo %LOCALAPPDATA%' 2>/dev/null | tr -d '\r')"
  wslpath -u "$win"
}
repair_ps1="\\\\wsl.localhost\\${distro}${repo_root//\//\\}\\scripts\\validation\\repair-multipass.ps1"

echo "=== repair-multipass.sh ==="
echo "  Accept the Windows Administrator (UAC) prompt."
echo ""

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "
  \$repair = '$repair_ps1'
  \$exe = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell.exe' }
  Start-Process -FilePath \$exe -Verb RunAs -Wait -ArgumentList @(
    '-NoProfile','-ExecutionPolicy','Bypass','-File',\$repair
  )
"

echo ""
log_win="$(win_localappdata)/Temp/rl-adaptive-dbs-multipass-repair/repair.log"
if [[ -f "$log_win" ]]; then
  echo "=== repair.log ==="
  tail -30 "$log_win"
else
  echo "Repair log not found at $log_win (check UAC was accepted)."
fi

echo ""
echo "=== multipass list ==="
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "& 'C:\Program Files\Multipass\bin\multipass.exe' list" 2>&1
