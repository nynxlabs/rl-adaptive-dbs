"""Eval tab data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.results import load_run_timeseries
from rl_adaptive_dbs.tui.eval_data import (
    cross_paper_warning,
    discover_eval_runs,
    eval_status_line,
    load_eval_context,
    p_beta_sparkline_data,
    select_eval_run,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark_results"
RUN_DIR = (
    FIXTURES
    / "mehregan_eval_smoke"
    / "runs"
    / "ddpg_paper_20260625-120030-c3d4"
)


def test_load_run_timeseries_rollout() -> None:
    payload = load_run_timeseries(RUN_DIR)
    assert payload is not None
    assert payload["p_beta"] == [92.0, 90.5, 89.2, 87.8, 85.0]


def test_discover_eval_runs_sorted_by_run_id_desc() -> None:
    _, suite, runs = load_eval_context(FIXTURES)
    assert suite is not None
    assert len(runs) == 2
    assert runs[0].run_id >= runs[1].run_id


def test_select_eval_run_attaches_timeseries() -> None:
    _, suite, runs = load_eval_context(FIXTURES)
    assert suite is not None
    run = select_eval_run(runs, "20260625-120030-c3d4")
    assert run is not None
    assert run.has_timeseries
    spark = p_beta_sparkline_data(run)
    assert spark == [92.0, 90.5, 89.2, 87.8, 85.0]


def test_eval_status_line() -> None:
    _, suite, runs = load_eval_context(FIXTURES)
    assert suite is not None
    run = select_eval_run(runs, runs[0].run_id)
    assert run is not None
    line = eval_status_line(suite, run, runs)
    assert "Eval:" in line
    assert "timeseries:" in line


def test_cross_paper_warning_hidden_for_mehregan() -> None:
    _, suite, _ = load_eval_context(FIXTURES)
    assert suite is not None
    assert cross_paper_warning(suite) is None


def test_discover_eval_runs_from_suite() -> None:
    _, suite, _ = load_eval_context(FIXTURES)
    assert suite is not None
    runs = discover_eval_runs(suite)
    assert len(runs) == 2
