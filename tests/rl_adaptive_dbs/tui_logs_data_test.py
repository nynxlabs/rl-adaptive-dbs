"""Logs tab data layer tests (no terminal required)."""

from __future__ import annotations

import json
from pathlib import Path

from rl_adaptive_dbs.tui.logs_data import (
    DEFAULT_TAIL_LINES,
    add_bookmark,
    bookmarks_file,
    discover_log_files,
    filter_log_files,
    format_log_line,
    is_bookmarked,
    load_bookmarks,
    log_file_rows,
    remove_bookmark,
    tail_lines,
    toggle_bookmark,
)
from rl_adaptive_dbs.run_log_meta import RunLogMeta, write_run_log_header

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "log_files"
TRAINING_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "training_artifacts"


def _missing_logs(tmp_path: Path) -> Path:
    return tmp_path / "missing_logs"


def test_discover_log_files_from_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifacts.mkdir()
    results.mkdir()
    source = FIXTURES / "sample.log"
    target = artifacts / "train.log"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    files = discover_log_files(results, artifacts, logs_dir=_missing_logs(tmp_path))
    assert len(files) == 1
    assert files[0].source == "artifacts"
    assert files[0].display_path == "train.log"


def test_discover_log_files_from_logs_dir(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    logs = tmp_path / "logs"
    artifacts.mkdir()
    results.mkdir()
    logs.mkdir()
    probe_log = logs / "fig4a-alphabet-sweep.log"
    probe_log.write_text("episode 1/10\n", encoding="utf-8")

    files = discover_log_files(results, artifacts, logs_dir=logs)
    assert len(files) == 1
    assert files[0].source == "logs"
    assert files[0].display_path == "fig4a-alphabet-sweep.log"


def test_discover_includes_training_fixture_logs() -> None:
    files = discover_log_files(
        TRAINING_FIXTURES.parent / "missing",
        TRAINING_FIXTURES,
        logs_dir=Path("/nonexistent/logs"),
    )
    paths = {item.display_path for item in files}
    assert "ddpg/train_paper_seed0.log" in paths
    assert "ddpg/paper/train_log.jsonl" in paths


def test_filter_log_files() -> None:
    files = discover_log_files(
        TRAINING_FIXTURES.parent / "missing",
        TRAINING_FIXTURES,
        logs_dir=Path("/nonexistent/logs"),
    )
    filtered = filter_log_files(files, "jsonl")
    assert filtered
    assert all("jsonl" in item.display_path.lower() for item in filtered)


def test_tail_lines_returns_last_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.log"
    path.write_text("\n".join(f"line {index}" for index in range(250)), encoding="utf-8")
    lines, error = tail_lines(path, max_lines=DEFAULT_TAIL_LINES)
    assert error is None
    assert len(lines) == DEFAULT_TAIL_LINES
    assert lines[0] == "line 50"
    assert lines[-1] == "line 249"


def test_tail_lines_missing_file(tmp_path: Path) -> None:
    lines, error = tail_lines(tmp_path / "missing.log")
    assert lines == []
    assert error is not None


def test_bookmarks_round_trip(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    log_path = FIXTURES / "events.jsonl"
    path = bookmarks_file(artifacts)

    add_bookmark(path, log_path)
    loaded = load_bookmarks(path)
    assert loaded == [log_path.resolve()]

    files = discover_log_files(
        tmp_path / "results",
        artifacts,
        logs_dir=_missing_logs(tmp_path),
    )
    assert files[0].is_bookmark is True
    assert files[0].path == log_path.resolve()


def test_bookmarks_persist_as_json(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    path = bookmarks_file(artifacts)
    log_path = FIXTURES / "sample.log"
    add_bookmark(path, log_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == [str(log_path.resolve())]


def test_remove_bookmark(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    log_path = FIXTURES / "sample.log"
    path = bookmarks_file(artifacts)
    add_bookmark(path, log_path)
    assert is_bookmarked(path, log_path)

    remove_bookmark(path, log_path)
    assert load_bookmarks(path) == []
    assert not is_bookmarked(path, log_path)


def test_toggle_bookmark(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    log_path = FIXTURES / "sample.log"
    path = bookmarks_file(artifacts)

    assert toggle_bookmark(path, log_path) is True
    assert is_bookmarked(path, log_path)
    assert toggle_bookmark(path, log_path) is False
    assert not is_bookmarked(path, log_path)
    assert load_bookmarks(path) == []


def test_format_log_line_color_markup() -> None:
    assert format_log_line("plain", color=False) == "plain"
    assert "[red]" in format_log_line("ERROR failed", color=True)
    assert "[yellow]" in format_log_line("WARNING slow", color=True)


def test_discover_log_files_parses_run_meta(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifacts.mkdir()
    results.mkdir()
    log_path = artifacts / "figures" / "run.log"
    log_path.parent.mkdir(parents=True)
    write_run_log_header(
        log_path,
        RunLogMeta(
            pid=999999,
            command="uv run python plot.py",
            started_at="2026-07-09T20:00:00+00:00",
            tmux_session="fig2a",
        ),
    )

    files = discover_log_files(
        results,
        artifacts,
        logs_dir=_missing_logs(tmp_path),
    )
    assert len(files) == 1
    item = files[0]
    assert item.run_meta is not None
    assert item.run_meta.tmux_session == "fig2a"
    assert item.run_state == "finished"
    rows = log_file_rows(files)
    assert rows[0][3] == "finished"


def test_filter_log_files_matches_tmux_session(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifacts.mkdir()
    results.mkdir()
    log_path = artifacts / "run.log"
    write_run_log_header(
        log_path,
        RunLogMeta(
            pid=1,
            command="test",
            started_at="2026-07-09T20:00:00+00:00",
            tmux_session="task174-train",
        ),
    )
    files = discover_log_files(
        results,
        artifacts,
        logs_dir=_missing_logs(tmp_path),
    )
    filtered = filter_log_files(files, "task174")
    assert len(filtered) == 1

