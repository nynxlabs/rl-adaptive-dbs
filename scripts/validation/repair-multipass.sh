#!/usr/bin/env bash
# Repair wedged Multipass on Windows (elevated). See scripts/validation/repair-multipass.ps1
set -euo pipefail
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/../.." && pwd)"
distro="${WSL_DISTRO_NAME:-Ubuntu}"
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
log_win="/mnt/c/Users/Devat/AppData/Local/Temp/rl-adaptive-dbs-multipass-repair/repair.log"
if [[ -f "$log_win" ]]; then
  echo "=== repair.log ==="
  tail -30 "$log_win"
else
  echo "Repair log not found at $log_win (check UAC was accepted)."
fi

echo ""
echo "=== multipass list ==="
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "& 'C:\Program Files\Multipass\bin\multipass.exe' list" 2>&1
