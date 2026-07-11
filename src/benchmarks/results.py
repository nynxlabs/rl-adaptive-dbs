"""Write benchmark results under ``results/`` ([benchmarking.md](../../docs/benchmarking.md) §6)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.schema import RunRecord, SuiteManifest


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    msg = f"object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def suite_output_dir(results_dir: Path, suite: SuiteManifest) -> Path:
    return results_dir / suite.suite_dir_name


def write_run_outputs(
    suite_dir: Path,
    record: RunRecord,
    *,
    write_timeseries: bool = True,
) -> Path:
    """Write ``config.json``, ``metrics.json``, and optional ``timeseries/`` for one run."""
    record.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(record.run_dir / "config.json", record.config)
    _write_json(record.run_dir / "metrics.json", record.metrics)
    if write_timeseries and record.timeseries:
        ts_dir = record.run_dir / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        for name, series in record.timeseries.items():
            _write_json(ts_dir / f"{name}.json", series)
    return record.run_dir


def build_suite_manifest_payload(
    suite: SuiteManifest,
    *,
    results_dir: Path,
    git_commit: str | None,
    planned_runs: int,
    completed_runs: int,
    started_at: str,
    finished_at: str | None = None,
    env_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": suite.name,
        "version": suite.version,
        "protocol": suite.protocol,
        "seeds": list(suite.seeds),
        "eval_steps": suite.eval_steps,
        "controllers": [
            {
                "controller": entry.controller,
                "variant": entry.variant,
                **({"checkpoint": str(entry.checkpoint)} if entry.checkpoint else {}),
                **({"adapter": entry.adapter} if entry.adapter is not None else {}),
            }
            for entry in suite.controllers
        ],
        "results_dir": str(results_dir),
        "planned_runs": planned_runs,
        "completed_runs": completed_runs,
        "started_at": started_at,
    }
    if suite.env_ref:
        payload["env_ref"] = suite.env_ref
    if suite.metrics:
        payload["metrics"] = list(suite.metrics)
    if suite.suite_path:
        payload["suite_path"] = str(suite.suite_path)
    if finished_at:
        payload["finished_at"] = finished_at
    if git_commit:
        payload["git"] = {"commit": git_commit}
    if env_snapshot:
        payload["env"] = env_snapshot
    return payload


def write_suite_manifest(suite_dir: Path, payload: dict[str, Any]) -> Path:
    path = suite_dir / "manifest.json"
    _write_json(path, payload)
    return path


def load_suite_manifest(suite_dir: Path) -> dict[str, Any]:
    path = suite_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


_TIMESERIES_CANDIDATES = ("rollout.json", "p_beta.json")


def load_run_timeseries(run_dir: Path) -> dict[str, Any] | None:
    """Load per-step series from ``timeseries/`` (``rollout.json`` preferred)."""
    ts_dir = run_dir / "timeseries"
    if not ts_dir.is_dir():
        return None
    for name in _TIMESERIES_CANDIDATES:
        path = ts_dir / name
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None
    for path in sorted(ts_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def list_suite_runs(suite_dir: Path) -> list[Path]:
    runs_dir = suite_dir / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(path for path in runs_dir.iterdir() if path.is_dir())


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
