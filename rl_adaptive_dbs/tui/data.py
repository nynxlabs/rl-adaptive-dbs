"""Textual TUI for browsing benchmark results (Phase 4 — Benchmarks tab)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.loader import SuiteResults, discover_suite_dirs, filter_runs, load_suite_results
from benchmarks.summary import render_summary_table


def refresh_suites(results_dir: Path) -> list[SuiteResults]:
    """Load all suite result trees under ``results_dir``."""
    return [load_suite_results(path) for path in discover_suite_dirs(results_dir)]


def select_suite(suites: list[SuiteResults], name: str | None) -> SuiteResults | None:
    if not suites:
        return None
    if name:
        for suite in suites:
            if suite.name == name:
                return suite
    return suites[0]


def suite_table_rows(suite: SuiteResults, *, query: str = "") -> list[tuple[str, ...]]:
    """Rows for the Benchmarks DataTable."""
    rows: list[tuple[str, ...]] = []
    for run in filter_runs(suite.runs, query):
        if run.error:
            rows.append(
                (
                    run.controller,
                    run.variant,
                    str(run.seed),
                    "ERR",
                    "ERR",
                    run.error[:24],
                    run.run_id[:20],
                )
            )
            continue
        reward = f"{run.reward_sum:.1f}" if suite.show_reward_sum else "n/a"
        rows.append(
            (
                run.controller,
                run.variant,
                str(run.seed),
                f"{run.p_beta_mean:.1f}",
                f"{run.p_beta_final:.1f}",
                reward,
                run.run_id[:20],
            )
        )
    return rows


def suite_status_line(suite: SuiteResults) -> str:
    planned = suite.planned_runs or suite.completed_runs
    seeds = f"{min(suite.seeds)}-{max(suite.seeds)}" if suite.seeds else "?"
    return (
        f"Benchmarks: {suite.name} v{suite.version}  "
        f"protocol: {suite.protocol or '?'}  seeds: {seeds}  "
        f"runs: {suite.completed_runs}/{planned}"
    )


def placeholder_tab_message(tab: str) -> str:
    return (
        f"{tab} tab — not implemented yet (Phase 4+).\n\n"
        "Use `uv run rl-dbs train` / `uv run rl-dbs benchmark` in another terminal."
    )


def ascii_fallback(results_dir: Path, *, width: int = 80) -> str:
    """Non-TTY fallback: print the latest suite summary table."""
    suites = refresh_suites(results_dir)
    suite = select_suite(suites, None)
    if suite is None:
        return f"No benchmark results under {results_dir}."
    return render_summary_table(suite, width=width)
