#!/usr/bin/env bash
# Internal worker for run_resilient.sh — writes run metadata, runs plot.py, appends exit footer.
set -euo pipefail

LOG="$1"
PIDFILE="$2"
REPO_ROOT="$3"
SEEDS="$4"
STEP_S="${5:-0.2}"
WINDOW_S="${6:-2}"
CPUS="$7"
TMUX_SESSION="${8:-}"

cd "$REPO_ROOT"

command_text="uv run python scripts/figures/papers/mehregan/2a/plot.py --seeds ${SEEDS} --step-s ${STEP_S} --window-s ${WINDOW_S}"

uv run python -m rl_adaptive_dbs.run_log_meta write-header \
    --log "$LOG" \
    --pid "$$" \
    --command "$command_text" \
    --tmux-session "${TMUX_SESSION:-}" \
    --pid-file "$PIDFILE" \
    --repo-root "$REPO_ROOT" \
    --cpus "$CPUS"

export PYTHONUNBUFFERED=1
set +e
taskset -c "$CPUS" uv run python scripts/figures/papers/mehregan/2a/plot.py \
    --seeds "$SEEDS" \
    --step-s "$STEP_S" \
    --window-s "$WINDOW_S"
exit_code=$?
set -e

uv run python -m rl_adaptive_dbs.run_log_meta write-exit --log "$LOG" --exit-code "$exit_code"
rm -f "$PIDFILE"
exit "$exit_code"
