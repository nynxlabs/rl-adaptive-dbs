#!/usr/bin/env bash
# Bootstrap a fresh Linux Multipass guest: apt + uv + git clone + validate-fresh.sh.
# Used by scripts/run-multipass-linux-validation.ps1 (always clone from GitHub).
set -euo pipefail

REPO_URL="${RL_ADAPTIVE_DBS_REPO_URL:-https://github.com/wilzen3476/rl-adaptive-dbs.git}"
REPO_DIR="${RL_ADAPTIVE_DBS_REPO_DIR:-$HOME/rl-adaptive-dbs}"

say() { printf '%s\n' "$*"; }

say "=== bootstrap-fresh-linux.sh ==="

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  say "Installing git + curl (apt)..."
  sudo apt-get update
  sudo apt-get install -y git curl ca-certificates
fi

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  say "Cloning $REPO_URL -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
bash scripts/validate-fresh.sh "$@"
