"""Benchmark results loader and summary tests."""

from __future__ import annotations

from pathlib import Path

from benchmarks.loader import filter_runs, load_suite_results
from benchmarks.summary import load_results_summary, render_summary_table, rows_to_csv

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark_results"
SUITE_DIR = FIXTURES / "mehregan_eval_smoke"


def test_load_suite_results_fixture() -> None:
    suite = load_suite_results(SUITE_DIR)
    assert suite.name == "mehregan_eval_smoke"
    assert suite.protocol == "mehregan"
    assert len(suite.runs) == 2
    assert suite.runs[0].p_beta_mean <= suite.runs[1].p_beta_mean


def test_filter_runs() -> None:
    suite = load_suite_results(SUITE_DIR)
    filtered = filter_runs(suite.runs, "ddpg")
    assert len(filtered) == 1
    assert filtered[0].controller == "ddpg"


def test_render_summary_table() -> None:
    suite = load_suite_results(SUITE_DIR)
    text = render_summary_table(suite, width=100)
    assert "mehregan_eval_smoke" in text
    assert "ddpg" in text
    assert "142.1" in text


def test_rows_to_csv() -> None:
    suite = load_suite_results(SUITE_DIR)
    csv_text = rows_to_csv(suite)
    assert "p_beta_mean" in csv_text
    assert "ddpg" in csv_text


def test_load_results_summary_by_name() -> None:
    suite = load_results_summary(FIXTURES, "mehregan_eval_smoke")
    assert suite.name == "mehregan_eval_smoke"
