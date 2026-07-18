#!/usr/bin/env bash
# Fresh-host validation for Phase 4 portability checks.
# Run inside Multipass Ubuntu (Linux) or Windows Sandbox (Git Bash) after git clone.
# See docs/development/fresh-validation.md.
set -euo pipefail

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_script_dir/.." && pwd)"
cd "$repo_root"

RUN_SETUP=1
LOG_FILE="${VALIDATION_LOG:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checks-only)
      RUN_SETUP=0
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    -h | --help)
      cat <<'EOF'
Usage: bash scripts/validate-fresh.sh [options]

Run project setup (optional) and print a copy-paste validation report.

Options:
  --checks-only   Skip scripts/setup.sh (deps already installed)
  --log-file PATH Append all output to PATH (also printed); or set VALIDATION_LOG
  -h, --help      Show this help

Typical flow (Multipass or Windows Sandbox):
  git clone https://github.com/wilzen3476/rl-adaptive-dbs.git
  cd rl-adaptive-dbs
  bash scripts/validate-fresh.sh

Windows host prerequisites (run on desktop PowerShell):
  pwsh -File scripts/check-windows-host.ps1
  pwsh -File scripts/refresh-multipass-catalog.ps1   # if Multipass launch fails (Admin)

See docs/development/fresh-validation.md (timing, troubleshooting).

Related:
  bash scripts/setup.sh --python-only --non-interactive   # install + pytest only
  bash scripts/setup.sh --python-only --non-interactive --validate  # + this report
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  {
    printf '=== validate-fresh.sh started %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u)"
  } >>"$LOG_FILE"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "validate-fresh.sh: uv not found — install https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ "$RUN_SETUP" -eq 1 ]]; then
  bash "$_script_dir/setup.sh" --python-only --non-interactive --skip-tests
else
  uv sync --group dev --group figures
fi

say_section() { printf '\n=== %s ===\n' "$*"; }

say_section "Import check"
uv run python -c "import envs; import controllers; print('ok')"

say_section "pytest (excluding matlab marker)"
uv run pytest -m "not matlab" -q

say_section "CLI smoke"
uv run rl-dbs info >/dev/null
uv run rl-dbs benchmark --suite-name mehregan_eval_smoke --dry-run >/dev/null

_git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
_uv_version="$(uv --version 2>/dev/null || echo unknown)"
_uname="$(uname -srmo 2>/dev/null || uname -a 2>/dev/null || echo unknown)"
_shell="${SHELL:-unknown}"
_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u)"

cat <<EOF

=== rl-adaptive-dbs fresh validation (paste into roadmap / issue) ===
date_utc:     $_date
platform:     $_uname
shell:        $_shell
git_sha:      $_git_sha
uv:           $_uv_version
setup:        $([[ "$RUN_SETUP" -eq 1 ]] && echo "setup.sh --python-only (skip-tests) + checks" || echo "checks-only")
pytest:       passed (-m "not matlab")
cli:          rl-dbs info + benchmark dry-run ok
macos:        deferred (no hardware)
=== end ===

EOF
