"""Load benchmark results from ``results/`` for TUI and summary tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.results import load_run_config, load_run_metrics, load_suite_manifest


@dataclass
class RunRow:
    """One completed benchmark run (metrics + config)."""

    run_dir: Path
    controller: str
    variant: str
    seed: int
    run_id: str
    p_beta_mean: float
    p_beta_final: float
    reward_sum: float
    stim_frequency_mean: float
    protocol: str
    error: str | None = None

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> RunRow:
        try:
            metrics = load_run_metrics(run_dir)
            config = load_run_config(run_dir)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            return cls(
                run_dir=run_dir,
                controller="?",
                variant="?",
                seed=-1,
                run_id=run_dir.name,
                p_beta_mean=float("nan"),
                p_beta_final=float("nan"),
                reward_sum=float("nan"),
                stim_frequency_mean=float("nan"),
                protocol="",
                error=str(exc),
            )
        return cls(
            run_dir=run_dir,
            controller=str(metrics.get("controller", config.get("controller", "?"))),
            variant=str(metrics.get("variant", config.get("variant", "?"))),
            seed=int(metrics.get("seed", config.get("seed", -1))),
            run_id=str(metrics.get("run_id", config.get("run_id", run_dir.name))),
            p_beta_mean=float(metrics.get("p_beta_mean", float("nan"))),
            p_beta_final=float(metrics.get("p_beta_final", float("nan"))),
            reward_sum=float(metrics.get("reward_sum", float("nan"))),
            stim_frequency_mean=float(metrics.get("stim_frequency_mean", float("nan"))),
            protocol=str(metrics.get("protocol", config.get("protocol", ""))),
        )


@dataclass
class SuiteResults:
    """All runs under ``results/<suite>/``."""

    suite_dir: Path
    name: str
    version: int
    protocol: str
    planned_runs: int
    completed_runs: int
    seeds: list[int] = field(default_factory=list)
    runs: list[RunRow] = field(default_factory=list)
    manifest_error: str | None = None

    @property
    def show_reward_sum(self) -> bool:
        if self.protocol in {"cross_paper", "cross_controller_plant"}:
            return False
        metrics = self._manifest_metrics()
        if metrics is not None and "reward_sum" not in metrics:
            return False
        return True

    def _manifest_metrics(self) -> list[str] | None:
        manifest_path = self.suite_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = payload.get("metrics")
        return list(raw) if raw else None


def discover_suite_dirs(results_dir: Path) -> list[Path]:
    """Return subdirs of ``results_dir`` that contain ``manifest.json``."""
    if not results_dir.is_dir():
        return []
    suites: list[Path] = []
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and (path / "manifest.json").is_file():
            suites.append(path)
    return suites


def load_suite_results(suite_dir: Path) -> SuiteResults:
    """Load manifest and all run metrics under ``suite_dir``."""
    name = suite_dir.name
    version = 1
    protocol = ""
    planned = 0
    completed = 0
    seeds: list[int] = []
    manifest_error: str | None = None

    try:
        manifest = load_suite_manifest(suite_dir)
        name = str(manifest.get("name", name))
        version = int(manifest.get("version", 1))
        protocol = str(manifest.get("protocol", ""))
        planned = int(manifest.get("planned_runs", 0))
        completed = int(manifest.get("completed_runs", 0))
        seeds = [int(s) for s in manifest.get("seeds", [])]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        manifest_error = str(exc)

    runs_dir = suite_dir / "runs"
    runs: list[RunRow] = []
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                runs.append(RunRow.from_run_dir(run_dir))

    runs.sort(key=lambda row: (row.p_beta_mean, row.controller, row.variant, row.seed))
    if completed == 0:
        completed = len(runs)

    return SuiteResults(
        suite_dir=suite_dir,
        name=name,
        version=version,
        protocol=protocol,
        planned_runs=planned,
        completed_runs=completed,
        seeds=seeds,
        runs=runs,
        manifest_error=manifest_error,
    )


def filter_runs(runs: list[RunRow], query: str) -> list[RunRow]:
    """Case-insensitive substring filter on controller or variant."""
    needle = query.strip().lower()
    if not needle:
        return runs
    return [
        row
        for row in runs
        if needle in row.controller.lower() or needle in row.variant.lower()
    ]
