#!/usr/bin/env bash
# Run on the main checkout after landing a figure ship from a worktree.
# Restores vault symlinks, copies tracker-linked PNGs, refreshes Report 3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

if grep -qE '/worktrees/' "${REPO_ROOT}/.git" 2>/dev/null; then
  echo "sync_vault_after_ship: run from the main checkout, not a linked worktree" >&2
  echo "  cd ~/bme/rl-adaptive-dbs && bash scripts/figures/papers/sync_vault_after_ship.sh" >&2
  exit 1
fi

HOOKS="${HOME}/setup/knowledge-base/githooks"
if [[ -x "${HOOKS}/restore-vault-md-symlinks.sh" ]]; then
  VAULT_MD_REPO_ROOT="$REPO_ROOT" bash "${HOOKS}/restore-vault-md-symlinks.sh"
else
  echo "warning: vault-md restore hook missing at ${HOOKS}" >&2
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/push_kb_images.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/update_report3.py
fi

echo "sync_vault_after_ship: vault replication images + Report 3 refreshed"
