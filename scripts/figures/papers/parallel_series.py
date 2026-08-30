"""Process-parallel helpers for multi-series paper panel scripts.

Each series/variant (plot line) can run in its own process with ``RL_DBS_MAX_THREADS=1``.
Use ``--parallel-series 0`` (default) to auto-split up to ``min(n_tasks, cpu_count)``;
``1`` keeps the legacy sequential loop; ``N > 1`` caps worker count.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from rl_adaptive_dbs.parallel_workers import effective_workers, run_in_parallel

T = TypeVar("T")
R = TypeVar("R")

PARALLEL_SERIES_HELP = (
    "Run independent series/variants in separate processes (each capped at 1 plant thread). "
    "0 = auto (min(task count, CPU count)); 1 = sequential; N > 1 caps workers."
)


def bootstrap_worker_threads() -> None:
    """Cap in-process Numba/OpenBLAS pools in pool workers."""
    from rl_adaptive_dbs.thread_limits import apply_max_threads

    apply_max_threads(1)


def default_parallel_series_count(n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    cpus = os.cpu_count() or 1
    return min(n_tasks, cpus)


def resolve_parallel_series(arg: int, n_tasks: int) -> int:
    """Map CLI ``--parallel-series`` to an effective worker count."""
    if n_tasks <= 0:
        return 1
    if arg <= 0:
        return default_parallel_series_count(n_tasks)
    return effective_workers(arg, n_tasks)


def add_parallel_series_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = 0,
) -> None:
    parser.add_argument(
        "--parallel-series",
        type=int,
        default=default,
        metavar="N",
        help=PARALLEL_SERIES_HELP,
    )


def run_series_parallel(
    items: list[T],
    worker: Callable[[T], R],
    parallel: int,
) -> list[R]:
    """Run *worker* on each item; process pool when ``parallel > 1``."""
    workers = resolve_parallel_series(parallel, len(items))
    if workers > 1:
        print(f"parallel-series: {workers} workers for {len(items)} tasks", flush=True)
    return run_in_parallel(items, worker, workers)


@dataclass(frozen=True)
class RavivarapuInferenceEvalJob:
    variant: str
    seed: int
    checkpoint: str
    max_steps: int
    carrier_hz: float
    dbs_burst_ms: float
    biomarker_window_s: float
    n_obs: int
    gumbel_seed_offset: int
    dbs_pulse_delay_ms: float | None = None


@dataclass(frozen=True)
class RavivarapuInferenceEvalResult:
    variant: str
    trace: list[float]
    actions: list[int]
    eval_meta: dict[str, Any]


def ravivarapu_inference_eval_worker(job: RavivarapuInferenceEvalJob) -> RavivarapuInferenceEvalResult:
    bootstrap_worker_threads()
    import numpy as np

    from controllers.sea_dbs.config import SEADBSConfig
    from controllers.sea_dbs.eval import evaluate

    kwargs: dict[str, Any] = {
        "max_steps": job.max_steps,
        "carrier_hz": job.carrier_hz,
        "action_mode": "gumbel",
        "dbs_burst_ms": job.dbs_burst_ms,
        "biomarker_window_s": job.biomarker_window_s,
        "n_obs": job.n_obs,
        "gumbel_seed_offset": job.gumbel_seed_offset,
    }
    if job.dbs_pulse_delay_ms is not None:
        kwargs["dbs_pulse_delay_ms"] = job.dbs_pulse_delay_ms
    payload = evaluate(
        Path(job.checkpoint),
        config=SEADBSConfig(variant=job.variant, seed=job.seed),
        **kwargs,
    )
    actions = payload["action_trajectories"][0]
    eval_meta = {
        "carrier_hz": payload["carrier_hz"],
        "dbs_burst_ms": payload["dbs_burst_ms"],
        "biomarker_window_s": payload.get("biomarker_window_s"),
        "n_obs": payload.get("n_obs"),
        "gumbel_seed_offset": job.gumbel_seed_offset,
        "n_psd_samples": payload["n_psd_samples"],
        "action_mode": payload["action_mode"],
        "stim_frac": float(np.mean(actions)),
    }
    if payload.get("dbs_pulse_delay_ms") is not None:
        eval_meta["dbs_pulse_delay_ms"] = payload["dbs_pulse_delay_ms"]
    return RavivarapuInferenceEvalResult(
        variant=job.variant,
        trace=list(payload["p_beta_trajectories"][0]),
        actions=[int(a) for a in actions],
        eval_meta=eval_meta,
    )


@dataclass(frozen=True)
class RavivarapuAblationEvalJob:
    variant: str
    seed: int
    checkpoint: str
    n_steps: int


def ravivarapu_ablation_eval_worker(job: RavivarapuAblationEvalJob) -> tuple[str, list[float]]:
    bootstrap_worker_threads()
    from controllers.sea_dbs.config import SEADBSConfig
    from controllers.sea_dbs.eval import evaluate_ablation_steps

    payload = evaluate_ablation_steps(
        Path(job.checkpoint),
        config=SEADBSConfig(variant=job.variant, seed=job.seed),
        n_steps=job.n_steps,
    )
    return job.variant, list(payload["p_beta_trajectories"][0])
