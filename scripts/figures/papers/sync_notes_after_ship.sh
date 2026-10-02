#!/usr/bin/env bash
# Run on the main checkout after landing a figure ship from a worktree.
# Copies tracker-linked PNGs into the notes folder and refreshes Report 3
# (see rl_adaptive_dbs.notes_export for where that folder comes from).
#
# If the checkout uses vault-md githooks, their restore hook does the same
# work and also relinks tracker symlinks. The hooks dir is the repo's
# core.hooksPath, or VAULT_MD_HOOKS_DIR when set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.git" ]] && grep -qE '/worktrees/' "${REPO_ROOT}/.git"; then
  main_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
  echo "sync_notes_after_ship: run from the main checkout, not a linked worktree" >&2
  echo "  cd ${main_root} && bash scripts/figures/papers/sync_notes_after_ship.sh" >&2
  exit 1
fi

hooks="${VAULT_MD_HOOKS_DIR:-}"
if [[ -z "$hooks" ]]; then
  hooks_path="$(git config --get core.hooksPath || true)"
  if [[ -n "$hooks_path" ]]; then
    [[ "$hooks_path" == /* ]] || hooks_path="${REPO_ROOT}/${hooks_path}"
    hooks="$hooks_path"
  fi
fi

if [[ -n "$hooks" && -f "${hooks}/restore-vault-md-symlinks.sh" ]]; then
  VAULT_MD_REPO_ROOT="$REPO_ROOT" bash "${hooks}/restore-vault-md-symlinks.sh"
else
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/export_notes_images.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/update_report3.py
fi

echo "sync_notes_after_ship: notes-folder replication images + Report 3 refreshed"
