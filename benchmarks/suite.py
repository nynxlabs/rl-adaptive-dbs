"""Load and validate benchmark suite YAML manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from benchmarks.schema import ControllerEntry, PlannedRun, SuiteManifest


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (or cwd) until ``suites/`` or ``pyproject.toml`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "suites").is_dir():
            return candidate
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def default_suites_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "suites"


def resolve_suite_path(path_or_name: str | Path, *, repo_root: Path | None = None) -> Path:
    path = Path(path_or_name)
    if path.is_file():
        return path.resolve()
    root = repo_root or find_repo_root()
    candidate = default_suites_dir(root) / f"{path.stem if path.suffix else path.name}.yaml"
    if candidate.is_file():
        return candidate.resolve()
    msg = f"suite manifest not found: {path_or_name}"
    raise FileNotFoundError(msg)


def _parse_controller_entry(raw: dict[str, Any]) -> ControllerEntry:
    checkpoint = raw.get("checkpoint")
    return ControllerEntry(
        controller=str(raw["controller"]),
        variant=str(raw["variant"]),
        checkpoint=Path(checkpoint) if checkpoint else None,
        train_seed=int(raw["train_seed"]) if raw.get("train_seed") is not None else None,
        adapter=raw.get("adapter"),
        metrics_extra=dict(raw.get("metrics_extra") or {}),
    )


def load_suite(path: str | Path, *, repo_root: Path | None = None) -> SuiteManifest:
    """Parse a suite YAML file into a ``SuiteManifest``."""
    suite_path = resolve_suite_path(path, repo_root=repo_root)
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"suite manifest must be a mapping: {suite_path}"
        raise ValueError(msg)

    controllers = tuple(
        _parse_controller_entry(entry)
        for entry in payload.get("controllers") or []
    )
    seeds_raw = payload.get("seeds") or [0]
    checkpoint_dir = payload.get("checkpoint_dir")
    return SuiteManifest(
        name=str(payload["name"]),
        version=int(payload.get("version", 1)),
        protocol=str(payload.get("protocol", payload["name"])),
        seeds=tuple(int(s) for s in seeds_raw),
        controllers=controllers,
        env_ref=str(payload["env_ref"]) if payload.get("env_ref") else None,
        metrics=tuple(str(m) for m in payload["metrics"]) if payload.get("metrics") else None,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
        train_seed=int(payload.get("train_seed", 0)),
        eval_steps=int(payload.get("eval_steps", 5)),
        suite_path=suite_path,
    )


def parse_controller_filter(raw: str | None) -> set[tuple[str, str]] | None:
    """Parse ``ddpg:paper,baseline:cdbs-130hz`` into controller/variant pairs."""
    if not raw:
        return None
    pairs: set[tuple[str, str]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            msg = f"invalid controller filter {item!r}; expected controller:variant"
            raise ValueError(msg)
        controller, variant = item.split(":", 1)
        pairs.add((controller.strip(), variant.strip()))
    return pairs


def expand_planned_runs(
    suite: SuiteManifest,
    *,
    seeds: tuple[int, ...] | None = None,
    controller_filter: set[tuple[str, str]] | None = None,
    repo_root: Path | None = None,
) -> list[PlannedRun]:
    """Expand suite manifest into individual runs."""
    root = repo_root or find_repo_root()
    active_seeds = seeds if seeds is not None else suite.seeds
    planned: list[PlannedRun] = []
    for entry in suite.controllers:
        if controller_filter is not None and (entry.controller, entry.variant) not in controller_filter:
            continue
        checkpoint = _resolve_checkpoint(entry, suite, repo_root=root)
        for seed in active_seeds:
            planned.append(
                PlannedRun(
                    controller=entry.controller,
                    variant=entry.variant,
                    seed=int(seed),
                    entry=entry,
                    checkpoint=checkpoint,
                )
            )
    return planned


def _resolve_checkpoint(
    entry: ControllerEntry,
    suite: SuiteManifest,
    *,
    repo_root: Path,
) -> Path | None:
    if entry.controller == "baseline":
        return None
    if entry.checkpoint is not None:
        path = entry.checkpoint
        return path if path.is_absolute() else (repo_root / path).resolve()
    train_seed = entry.train_seed if entry.train_seed is not None else suite.train_seed
    checkpoint_dir = suite.checkpoint_dir or Path("artifacts/ddpg")
    rel = checkpoint_dir / f"{entry.variant}_train{train_seed}.pt"
    return (repo_root / rel).resolve() if not rel.is_absolute() else rel.resolve()
