"""Benchmark suite schema ([benchmarking.md](../docs/benchmarking.md) §5–§6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ControllerEntry:
    controller: str
    variant: str
    checkpoint: Path | None = None
    train_seed: int | None = None
    adapter: bool | None = None
    metrics_extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteManifest:
    name: str
    version: int
    protocol: str
    seeds: tuple[int, ...]
    controllers: tuple[ControllerEntry, ...]
    env_ref: str | None = None
    metrics: tuple[str, ...] | None = None
    checkpoint_dir: Path | None = None
    train_seed: int = 0
    eval_steps: int = 5
    suite_path: Path | None = None

    @property
    def suite_dir_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class PlannedRun:
    """One (controller, variant, seed) execution."""

    controller: str
    variant: str
    seed: int
    entry: ControllerEntry
    checkpoint: Path | None = None


@dataclass
class RunRecord:
    planned: PlannedRun
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    config: dict[str, Any]
    timeseries: dict[str, list[Any]] | None = None
    error: str | None = None
