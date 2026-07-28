#!/usr/bin/env bash
# Install Multipass and/or Windows Sandbox on the Windows desktop (Admin UAC).
# Run from WSL:
#
#   bash scripts/validation/install-fresh-validation-host.sh
#   bash scripts/validation/install-fresh-validation-host.sh --sandbox
#   bash scripts/validation/install-fresh-validation-host.sh --multipass
#   bash scripts/validation/install-fresh-validation-host.sh --check
#
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/../.." && pwd)"

install_sandbox=0
install_multipass=0
check_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox) install_sandbox=1; shift ;;
    --multipass) install_multipass=1; shift ;;
    --check) check_only=1; shift ;;
    -h | --help)
      cat <<'EOF'
Usage: bash scripts/validation/install-fresh-validation-host.sh [options]

Install Windows Sandbox and/or Multipass on the desktop host (Administrator).

Options:
  --sandbox     Enable Windows Sandbox only
  --multipass   Install Multipass only
  --check       Run check-windows-host.ps1 (no install)
  -h, --help    Show this help

Default (no flags): install both. Accept the UAC prompt on Windows.

After install: bash scripts/validation/install-fresh-validation-host.sh --check
See docs/development/fresh-validation.md
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

distro="${WSL_DISTRO_NAME:-Ubuntu}"
repo_unc="\\\\wsl.localhost\\${distro}${repo_root//\//\\}"
install_ps1="${repo_unc}\\scripts\\validation\\install-fresh-validation-host.ps1"
check_ps1="${repo_unc}\\scripts\\validation\\check-windows-host.ps1"

pwsh='pwsh'
if ! command -v pwsh >/dev/null 2>&1; then
  pwsh='/mnt/c/Program Files/PowerShell/7/pwsh.exe'
  [[ -x "$pwsh" ]] || pwsh='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
fi

run_pwsh() {
  if [[ "$pwsh" == /* ]]; then
    "$pwsh" -NoProfile -ExecutionPolicy Bypass "$@"
  else
    pwsh -NoProfile -ExecutionPolicy Bypass "$@"
  fi
}

if [[ "$check_only" -eq 1 ]]; then
  run_pwsh -File "$check_ps1"
  exit $?
fi

ps_flags=''
if [[ "$install_sandbox" -eq 1 && "$install_multipass" -eq 0 ]]; then
  ps_flags='-Sandbox'
elif [[ "$install_multipass" -eq 1 && "$install_sandbox" -eq 0 ]]; then
  ps_flags='-Multipass'
fi

echo "=== install-fresh-validation-host.sh ==="
echo "  WSL repo: $repo_root"
echo "  Accept the Windows Administrator (UAC) prompt."
echo ""

# Elevated install on Windows desktop
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command "
  \$install = '$install_ps1'
  \$argList = @('-NoProfile','-ExecutionPolicy','Bypass','-File',\$install)
  if ('$ps_flags' -eq '-Sandbox') { \$argList += '-Sandbox' }
  if ('$ps_flags' -eq '-Multipass') { \$argList += '-Multipass' }
  \$exe = 'pwsh'
  if (-not (Get-Command pwsh -ErrorAction SilentlyContinue)) {
    \$exe = 'C:\Program Files\PowerShell\7\pwsh.exe'
    if (-not (Test-Path \$exe)) { \$exe = 'powershell.exe' }
  }
  Start-Process -FilePath \$exe -Verb RunAs -Wait -ArgumentList \$argList
"

echo ""
echo "Running host check..."
run_pwsh -File "$check_ps1"
