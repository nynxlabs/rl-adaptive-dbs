#!/usr/bin/env bash
# Project setup: Python (uv) + optional MATLAB. See docs/setup.md.
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/.." && pwd)"
cd "$repo_root"

WITH_MATLAB=""
SKIP_TESTS=0
INTERACTIVE=1
RUN_VALIDATE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Install Python dependencies, verify imports, run fast pytest, and optionally
configure MATLAB for the Kumaravelu plant bridge.

Options:
  --python-only    Skip MATLAB (default in non-interactive mode)
  --with-matlab    Run scripts/matlab/setup.sh (interactive MATLAB flow)
  --skip-tests     Skip pytest after Python setup
  --validate       After setup, run scripts/validate-fresh.sh --checks-only
  --non-interactive
                   No prompts; implies --python-only unless --with-matlab
  -h, --help       Show this help

Examples:
  bash scripts/setup.sh
  bash scripts/setup.sh --python-only
  bash scripts/setup.sh --with-matlab
  bash scripts/setup.sh --python-only --non-interactive --validate

Fresh Multipass / Sandbox hosts: bash scripts/validate-fresh.sh
Windows host prerequisites: pwsh -File scripts/check-windows-host.ps1
EOF
}

say() { printf '%s\n' "$*"; }
ask_yn() {
  local prompt="$1" default="${2:-y}"
  local reply
  read -r -p "$prompt [y/n] (default $default): " reply
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy] ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-only | --no-matlab)
      WITH_MATLAB=0
      shift
      ;;
    --with-matlab | --matlab)
      WITH_MATLAB=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --validate)
      RUN_VALIDATE=1
      shift
      ;;
    --non-interactive)
      INTERACTIVE=0
      [[ -z "$WITH_MATLAB" ]] && WITH_MATLAB=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      say "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

say "=== rl-adaptive-dbs setup ==="
say "Repo: $repo_root"
say "Guide: docs/setup.md"
say ""

if ! command -v uv >/dev/null 2>&1; then
  say "uv is required. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

say "=== Python dependencies (uv sync --all-groups) ==="
uv sync --all-groups
say ""

say "=== Import check ==="
uv run python -c "import envs; import controllers; print('ok')"
say ""

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  say "=== pytest (excluding matlab marker) ==="
  uv run pytest -m "not matlab" -q
  say ""
fi

if [[ -z "$WITH_MATLAB" ]]; then
  if [[ "$INTERACTIVE" -eq 1 ]] && [[ -t 0 ]]; then
    if ask_yn "Run MATLAB setup (plant bridge)?" n; then
      WITH_MATLAB=1
    else
      WITH_MATLAB=0
    fi
  else
    WITH_MATLAB=0
  fi
fi

if [[ "$WITH_MATLAB" -eq 1 ]]; then
  say "=== MATLAB setup ==="
  if [[ "$INTERACTIVE" -eq 1 ]] && [[ -t 0 ]]; then
    bash "$_script_dir/matlab/setup.sh"
  elif [[ -n "${MATLAB_ROOT:-}" ]]; then
    bash "$_script_dir/matlab/verify.sh"
  else
    say "MATLAB_ROOT not set — run interactively: bash scripts/matlab/setup.sh"
    exit 1
  fi
  say ""
fi

if [[ "$RUN_VALIDATE" -eq 1 ]]; then
  say "=== extended validation (validate-fresh.sh) ==="
  bash "$_script_dir/validate-fresh.sh" --checks-only
  say ""
fi

say "=== setup complete ==="
if [[ "$RUN_VALIDATE" -eq 0 ]]; then
  say "Fresh-host report: bash scripts/validate-fresh.sh"
fi
say "Next: docs/setup.md §5 (day-to-day commands)"
