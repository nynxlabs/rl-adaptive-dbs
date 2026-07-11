"""Eval rollout loader for the TUI Eval tab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.loader import RunRow, SuiteResults
from benchmarks.results import load_run_timeseries
from rl_adaptive_dbs.tui.data import refresh_suites, select_suite

MAX_SPARKLINE_POINTS = 10_000


@dataclass
class EvalRun:
    """One benchmark run with optional cached timeseries."""

    row: RunRow
    timeseries: dict[str, Any] | None = None
    timeseries_mtime: float = 0.0

    @property
    def run_id(self) -> str:
        return self.row.run_id

    @property
    def has_timeseries(self) -> bool:
        return self.timeseries is not None and bool(self.timeseries.get("p_beta"))


def discover_eval_runs(suite: SuiteResults) -> list[EvalRun]:
    """Return suite runs sorted by ``run_id`` descending (newest first)."""
    rows = sorted(suite.runs, key=lambda row: row.run_id, reverse=True)
    return [EvalRun(row=row) for row in rows]


def attach_timeseries(run: EvalRun) -> EvalRun:
    """Load timeseries from disk when the file changed."""
    ts_path = _timeseries_path(run.row.run_dir)
    if ts_path is None:
        run.timeseries = None
        run.timeseries_mtime = 0.0
        return run
    try:
        mtime = ts_path.stat().st_mtime
    except OSError:
        run.timeseries = None
        run.timeseries_mtime = 0.0
        return run
    if run.timeseries is not None and mtime == run.timeseries_mtime:
        return run
    run.timeseries = load_run_timeseries(run.row.run_dir)
    run.timeseries_mtime = mtime
    return run


def _timeseries_path(run_dir: Path) -> Path | None:
    ts_dir = run_dir / "timeseries"
    if not ts_dir.is_dir():
        return None
    for name in ("rollout.json", "p_beta.json"):
        path = ts_dir / name
        if path.is_file():
            return path
    for path in sorted(ts_dir.glob("*.json")):
        if path.is_file():
            return path
    return None


def select_eval_run(runs: list[EvalRun], run_id: str | None) -> EvalRun | None:
    if not runs:
        return None
    if run_id:
        for run in runs:
            if run.run_id == run_id:
                return attach_timeseries(run)
    return attach_timeseries(runs[0])


def cycle_eval_run_id(
    runs: list[EvalRun],
    current: str | None,
    delta: int,
) -> str | None:
    if not runs:
        return None
    ids = [run.run_id for run in runs]
    if current not in ids:
        return ids[0]
    index = (ids.index(current) + delta) % len(ids)
    return ids[index]


def p_beta_sparkline_data(
    run: EvalRun,
    *,
    max_points: int = MAX_SPARKLINE_POINTS,
) -> list[float]:
    series = run.timeseries or {}
    values = [float(x) for x in series.get("p_beta", [])]
    if len(values) <= max_points:
        return values
    return values[-max_points:]


def eval_run_table_rows(
    runs: list[EvalRun],
    *,
    show_reward: bool,
) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for run in runs:
        row = run.row
        if row.error:
            rows.append(
                (
                    row.controller,
                    row.variant,
                    str(row.seed),
                    "ERR",
                    "ERR",
                    row.run_id[:20],
                )
            )
            continue
        reward = f"{row.reward_sum:.1f}" if show_reward else "n/a"
        rows.append(
            (
                row.controller,
                row.variant,
                str(row.seed),
                f"{row.p_beta_mean:.1f}",
                reward,
                row.run_id[:20],
            )
        )
    return rows


def eval_status_line(
    suite: SuiteResults,
    run: EvalRun,
    runs: list[EvalRun] | None = None,
) -> str:
    picker = ""
    if runs and len(runs) > 1 and run.run_id in [item.run_id for item in runs]:
        index = [item.run_id for item in runs].index(run.run_id) + 1
        picker = f"  [{index}/{len(runs)}]"
    switch = "  [ and ] run" if runs and len(runs) > 1 else ""
    ts_note = "  timeseries: yes" if run.has_timeseries else "  timeseries: no"
    return (
        f"Eval: {suite.name} v{suite.version}{picker}{switch}  "
        f"protocol: {suite.protocol or '?'}  "
        f"{run.row.controller}/{run.row.variant}  seed: {run.row.seed}{ts_note}"
    )


def eval_summary_lines(run: EvalRun, *, show_reward: bool) -> list[str]:
    row = run.row
    if row.error:
        return [f"error: {row.error}"]
    lines = [
        f"run_id: {row.run_id}",
        f"P_beta mean: {row.p_beta_mean:.1f}",
        f"P_beta final: {row.p_beta_final:.1f}",
        f"stim frequency mean: {row.stim_frequency_mean:.1f} Hz",
    ]
    if show_reward:
        lines.insert(3, f"reward sum: {row.reward_sum:.1f}")
    else:
        lines.insert(3, "reward sum: n/a (not comparable across protocols)")
    if run.has_timeseries:
        points = len(run.timeseries.get("p_beta", [])) if run.timeseries else 0
        lines.append(f"timeseries points: {points}")
    return lines


def cross_paper_warning(suite: SuiteResults) -> str | None:
    if suite.show_reward_sum:
        return None
    return (
        "Cross-paper suite: reward_sum is not comparable across controllers. "
        "Compare P_beta and stim frequency only (see benchmarking.md §3.3)."
    )


def eval_empty_message(results_dir: Path) -> str:
    return (
        f"No benchmark runs under {results_dir}/.\n\n"
        "Run `uv run rl-dbs benchmark ...` or `uv run rl-dbs eval ...` "
        "to produce results with metrics.json (and optional timeseries/)."
    )


def load_eval_context(
    results_dir: Path,
    *,
    suite_name: str | None = None,
) -> tuple[list[SuiteResults], SuiteResults | None, list[EvalRun]]:
    suites = refresh_suites(results_dir)
    suite = select_suite(suites, suite_name)
    if suite is None:
        return suites, None, []
    return suites, suite, discover_eval_runs(suite)
