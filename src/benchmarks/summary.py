"""Benchmark comparison tables (stdout / CSV)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from benchmarks.loader import RunRow, SuiteResults, discover_suite_dirs, load_suite_results


def _fmt_float(value: float, *, width: int = 9, na: str = "n/a") -> str:
    if value != value:  # NaN
        return na.rjust(width)
    return f"{value:>{width}.1f}"


def _truncate_run_id(run_id: str, width: int = 20) -> str:
    if len(run_id) <= width:
        return run_id.ljust(width)
    keep = width - 1
    left = keep // 2
    right = keep - left
    return f"{run_id[:left]}…{run_id[-right:]}".ljust(width)


def render_summary_table(
    suite: SuiteResults,
    *,
    include_reward: bool | None = None,
    width: int = 80,
) -> str:
    """ASCII comparison table for terminal output."""
    show_reward = suite.show_reward_sum if include_reward is None else include_reward
    narrow = width < 100

    if narrow or not show_reward:
        headers = ["controller", "variant", "seed", "p_beta_mu", "p_beta_fin", "run_id"]
    else:
        headers = [
            "controller",
            "variant",
            "seed",
            "p_beta_mu",
            "p_beta_fin",
            "reward_sum",
            "stim_freq",
            "run_id",
        ]

    lines: list[str] = [
        f"Benchmarks: {suite.name} v{suite.version}  protocol: {suite.protocol or '?'}",
        f"runs: {suite.completed_runs}/{suite.planned_runs or suite.completed_runs}",
        "",
        "  ".join(h.ljust(12) if h != "run_id" else h.ljust(20) for h in headers),
        "-" * min(width, 96),
    ]

    for row in suite.runs:
        if row.error:
            lines.append(f"ERROR {row.run_dir.name}: {row.error}")
            continue
        cells = [
            row.controller.ljust(12),
            row.variant.ljust(12),
            str(row.seed).rjust(4).ljust(12),
            _fmt_float(row.p_beta_mean, width=8).ljust(12),
            _fmt_float(row.p_beta_final, width=8).ljust(12),
        ]
        if not narrow and show_reward:
            cells.append(_fmt_float(row.reward_sum, width=8).ljust(12))
            cells.append(_fmt_float(row.stim_frequency_mean, width=8).ljust(12))
        cells.append(_truncate_run_id(row.run_id, 20))
        lines.append("  ".join(cells))

    if not suite.runs:
        lines.append("(no runs found)")
    return "\n".join(lines)


def rows_to_csv(suite: SuiteResults, *, include_reward: bool | None = None) -> str:
    """CSV string for one suite."""
    show_reward = suite.show_reward_sum if include_reward is None else include_reward
    buffer = io.StringIO()
    fieldnames = [
        "controller",
        "variant",
        "seed",
        "p_beta_mean",
        "p_beta_final",
        "stim_frequency_mean",
        "run_id",
        "protocol",
    ]
    if show_reward:
        fieldnames.insert(5, "reward_sum")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in suite.runs:
        if row.error:
            continue
        payload: dict[str, Any] = {
            "controller": row.controller,
            "variant": row.variant,
            "seed": row.seed,
            "p_beta_mean": row.p_beta_mean,
            "p_beta_final": row.p_beta_final,
            "stim_frequency_mean": row.stim_frequency_mean,
            "run_id": row.run_id,
            "protocol": row.protocol,
        }
        if show_reward:
            payload["reward_sum"] = row.reward_sum
        writer.writerow(payload)
    return buffer.getvalue()


def write_summary_csv(path: Path, suite: SuiteResults) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows_to_csv(suite), encoding="utf-8")
    return path


def load_results_summary(
    results_dir: Path,
    suite_name: str | None = None,
) -> SuiteResults:
    """Load one suite by name or the most recently modified suite dir."""
    if suite_name:
        suite_dir = results_dir / suite_name
        if not (suite_dir / "manifest.json").is_file():
            msg = f"no manifest at {suite_dir / 'manifest.json'}"
            raise FileNotFoundError(msg)
        return load_suite_results(suite_dir)

    suites = discover_suite_dirs(results_dir)
    if not suites:
        msg = f"no benchmark suites under {results_dir}"
        raise FileNotFoundError(msg)
    suites.sort(key=lambda p: (p / "manifest.json").stat().st_mtime, reverse=True)
    return load_suite_results(suites[0])
