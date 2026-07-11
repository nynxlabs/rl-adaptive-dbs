"""Tests for run log metadata markers."""

from __future__ import annotations

import os
from pathlib import Path

from rl_adaptive_dbs.run_log_meta import (
    RunLogMeta,
    format_run_log_header,
    format_run_log_exit,
    parse_run_log_header_text,
    parse_run_log_exit_text,
    read_run_log_markers,
    run_state_for_log,
    run_status_label,
    write_run_log_header,
    write_run_log_exit,
)


def test_run_log_header_round_trip() -> None:
    meta = RunLogMeta(
        pid=12345,
        command="uv run python plot.py",
        started_at="2026-07-09T20:00:00+00:00",
        tmux_session="fig2a",
        pid_file="artifacts/figures/papers/1/2a/run.pid",
        repo_root="/tmp/repo",
        cpus="0-2",
    )
    text = format_run_log_header(meta) + "simulating seed 0...\n"
    parsed = parse_run_log_header_text(text)
    assert parsed is not None
    assert parsed.pid == 12345
    assert parsed.tmux_session == "fig2a"
    assert parsed.command == "uv run python plot.py"


def test_run_log_exit_and_state(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    meta = RunLogMeta(
        pid=os.getpid(),
        command="uv run python plot.py",
        started_at="2026-07-09T20:00:00+00:00",
        tmux_session="fig2a",
    )
    write_run_log_header(log_path, meta)
    log_path.open("a", encoding="utf-8").write("working...\n")

    assert run_state_for_log(log_path) == "running"
    assert "running" in run_status_label(log_path)
    assert "tmux:fig2a" in run_status_label(log_path)

    write_run_log_exit(log_path, exit_code=0)
    meta2, exit_info = read_run_log_markers(log_path)
    assert meta2 is not None
    assert exit_info is not None
    assert exit_info.exit_code == 0
    assert run_state_for_log(log_path) == "finished"


def test_failed_exit_state(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    write_run_log_header(
        log_path,
        RunLogMeta(pid=1, command="test", started_at="2026-07-09T20:00:00+00:00"),
    )
    write_run_log_exit(log_path, exit_code=2)
    assert run_state_for_log(log_path) == "failed"
    assert "failed (exit 2)" in run_status_label(log_path)
