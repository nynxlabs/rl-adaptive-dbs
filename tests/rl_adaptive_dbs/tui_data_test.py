"""TUI data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.tui.data import refresh_suites, suite_table_rows, suite_status_line

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "results"


def test_refresh_suites() -> None:
    suites = refresh_suites(FIXTURES)
    assert len(suites) == 1
    assert suites[0].name == "mehregan_eval_smoke"


def test_suite_table_rows() -> None:
    suites = refresh_suites(FIXTURES)
    rows = suite_table_rows(suites[0])
    assert len(rows) == 2
    assert rows[0][0] in {"baseline", "ddpg"}


def test_suite_status_line() -> None:
    suites = refresh_suites(FIXTURES)
    line = suite_status_line(suites[0])
    assert "mehregan_eval_smoke" in line
    assert "runs:" in line
