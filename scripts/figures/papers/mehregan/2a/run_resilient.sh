#!/usr/bin/env bash
# Detached Fig 2a simulation — survives Cursor agent / terminal teardown.
#
# Uses setsid + nohup (new session, no hangup) instead of tmux so a tmux-server
# restart does not kill the plant run. Pin to 3 CPUs with taskset; thread pools
# stay at library defaults.
#
#   bash scripts/figures/papers/mehregan/2a/run_resilient.sh start
#   bash scripts/figures/papers/mehregan/2a/run_resilient.sh status
#   bash scripts/figures/papers/mehregan/2a/run_resilient.sh stop
#
# Logs land under artifacts/figures/papers/mehregan/2a/run.log (TUI Logs tab scans
# artifacts/ recursively). The worker writes an ``# rl-dbs-run-meta`` header
# (pid, tmux session, command, …) so the TUI can show live run state while
# tailing. On start we also bookmark that path so it stays pinned.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_DIR="artifacts/figures/papers/mehregan/2a"
LOG="${FIG2A_LOG:-$REPO_ROOT/$ARTIFACT_DIR/run.log}"
PIDFILE="${FIG2A_PIDFILE:-$REPO_ROOT/$ARTIFACT_DIR/run.pid}"
SEEDS="${FIG2A_SEEDS:-0}"
STEP_S="${FIG2A_STEP_S:-0.2}"
WINDOW_S="${FIG2A_WINDOW_S:-2}"
CPUS="${FIG2A_CPUS:-0-2}"
WORKER="$REPO_ROOT/scripts/figures/papers/mehregan/2a/run_worker.sh"

detect_tmux_session() {
    if [[ -n "${FIG2A_TMUX:-}" ]]; then
        printf '%s' "$FIG2A_TMUX"
        return
    fi
    if [[ -n "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
        tmux display-message -p '#S' 2>/dev/null || true
    fi
}

bookmark_log_for_tui() {
    uv run python -c "
from pathlib import Path
from rl_adaptive_dbs.tui.logs_data import add_bookmark, bookmarks_file

repo = Path('${REPO_ROOT}')
log = Path('${LOG}').resolve()
add_bookmark(bookmarks_file(repo / 'artifacts'), log)
print(f'bookmarked for TUI Logs tab: {log}')
"
}

cmd_start() {
    if [[ -f "$PIDFILE" ]]; then
        local old_pid
        old_pid="$(cat "$PIDFILE")"
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "already running (pid $old_pid); log: $LOG"
            exit 1
        fi
        rm -f "$PIDFILE"
    fi

    mkdir -p "$REPO_ROOT/$ARTIFACT_DIR"
    : >"$LOG"

    local tmux_session
    tmux_session="$(detect_tmux_session)"

    chmod +x "$WORKER"
    setsid nohup "$WORKER" \
        "$LOG" \
        "$PIDFILE" \
        "$REPO_ROOT" \
        "$SEEDS" \
        "$STEP_S" \
        "$WINDOW_S" \
        "$CPUS" \
        "$tmux_session" \
        >>"$LOG" 2>&1 < /dev/null &
    echo $! >"$PIDFILE"
    bookmark_log_for_tui
    echo "started pid $(cat "$PIDFILE")"
    echo "log: $LOG"
    if [[ -n "$tmux_session" ]]; then
        echo "tmux: $tmux_session  (attach: tmux attach -t $tmux_session)"
    fi
    echo "cpus: $CPUS  step: ${STEP_S}s  window: ${WINDOW_S}s  seeds: $SEEDS"
    echo "check: bash scripts/figures/papers/mehregan/2a/run_resilient.sh status"
}

cmd_status() {
    if [[ ! -f "$PIDFILE" ]]; then
        echo "not running (no pid file)"
        tail -3 "$LOG" 2>/dev/null || true
        exit 0
    fi
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "running pid $pid"
        ps -p "$pid" -o pid,etime,pcpu,pmem,cmd --no-headers 2>/dev/null || true
    else
        echo "not running (stale pid $pid)"
        if grep -q '# rl-dbs-run-exit:' "$LOG" 2>/dev/null; then
            echo "finished (see log footer)"
        fi
    fi
    echo "--- log tail ---"
    tail -5 "$LOG" 2>/dev/null || true
}

cmd_stop() {
    if [[ ! -f "$PIDFILE" ]]; then
        echo "no pid file"
        exit 0
    fi
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        echo "stopped pid $pid"
    else
        echo "pid $pid not running"
    fi
    rm -f "$PIDFILE"
}

case "${1:-start}" in
    start) cmd_start ;;
    status) cmd_status ;;
    stop) cmd_stop ;;
    *)
        echo "usage: $0 {start|status|stop}" >&2
        exit 2
        ;;
esac
