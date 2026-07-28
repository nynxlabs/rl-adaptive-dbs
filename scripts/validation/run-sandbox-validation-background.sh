#!/usr/bin/env bash
# Launch Windows Sandbox validation in the background (from WSL).
#   bash scripts/validation/run-sandbox-validation-background.sh          # -Clone (recommended)
#   bash scripts/validation/run-sandbox-validation-background.sh --mapped # WSL tree mapped (dev only)
#
# Clone mode stages scripts + .validation-logs onto NTFS (LOCALAPPDATA) so Sandbox
# folder maps work when \\wsl.localhost\... is flaky.
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/../.." && pwd)"
log_dir="$repo_root/.validation-logs"
launcher_log="$log_dir/sandbox-launcher.log"

clone_flag='-Clone'
repo_path_arg=''
if [[ "${1:-}" == '--mapped' ]]; then
  clone_flag=''
elif [[ "${1:-}" == '--no-stage' ]]; then
  : # use WSL UNC paths (legacy)
else
  stage_root="${LOCALAPPDATA:-/mnt/c/Users/Devat/AppData/Local}/rl-adaptive-dbs-validation"
  # LOCALAPPDATA is unset in WSL; use desktop profile path.
  if [[ "$stage_root" == /mnt/* ]]; then
    stage_root="/mnt/c/Users/Devat/AppData/Local/rl-adaptive-dbs-validation"
  fi
  mkdir -p "$stage_root/.validation-logs/cache" "$stage_root/scripts"
  rsync -a --delete "$repo_root/scripts/" "$stage_root/scripts/"
  # Git installer only — repo shallow clone + uv cache are built on NTFS by the launcher.
  if compgen -G "$log_dir/cache/Git-"*.exe >/dev/null; then
    rsync -a "$log_dir"/cache/Git-*.exe "$stage_root/.validation-logs/cache/"
  fi
  rsync -a "$log_dir/sandbox-launcher.log" "$stage_root/.validation-logs/" 2>/dev/null || true
  repo_path_arg="$(wslpath -w "$stage_root")"
  echo "Staged validation files on NTFS: $stage_root" >>"$launcher_log"
fi

mkdir -p "$log_dir"
rm -f "$log_dir/sandbox.log"
pwsh='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'

if [[ -n "$repo_path_arg" ]]; then
  script_win="$(wslpath -w "$stage_root/scripts/validation/launch-windows-sandbox-validation.ps1")"
else
  script_win="$(wslpath -w "$_script_dir/launch-windows-sandbox-validation.ps1")"
fi

"$pwsh" -NoProfile -Command \
  "Stop-Process -Name WindowsSandbox -Force -ErrorAction SilentlyContinue" || true

echo "=== sandbox launcher $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"$launcher_log"

launch_args=(-NoProfile -ExecutionPolicy Bypass -File "$script_win")
[[ -n "$clone_flag" ]] && launch_args+=("$clone_flag")
[[ -n "$repo_path_arg" ]] && launch_args+=(-RepoPath "$repo_path_arg")

nohup "$pwsh" "${launch_args[@]}" >>"$launcher_log" 2>&1 &

# Mirror sandbox.log from NTFS stage back into the WSL repo for tail -f.
if [[ -n "${stage_root:-}" ]]; then
  (
    while "$pwsh" -NoProfile -Command \
      'Get-Process WindowsSandbox -ErrorAction SilentlyContinue' >/dev/null 2>&1; do
      if [[ -f "$stage_root/.validation-logs/sandbox.log" ]]; then
        cp -f "$stage_root/.validation-logs/sandbox.log" "$log_dir/sandbox.log" 2>/dev/null || true
      fi
      sleep 5
    done
    if [[ -f "$stage_root/.validation-logs/sandbox.log" ]]; then
      cp -f "$stage_root/.validation-logs/sandbox.log" "$log_dir/sandbox.log" 2>/dev/null || true
    fi
  ) &
  sync_pid=$!
  echo "  Log sync PID: $sync_pid (NTFS -> WSL until Sandbox exits)"
fi

echo "Sandbox validation launched in background (PID $!)."
echo "  Validation log: $log_dir/sandbox.log"
echo "  Launcher log:   $launcher_log"
echo "  tail -f $log_dir/sandbox.log"
[[ -n "${stage_root:-}" ]] && echo "  NTFS stage:     $stage_root/.validation-logs/sandbox.log"
