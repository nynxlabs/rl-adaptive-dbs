"""Process-pool workers for independent MATLAB-backed seeds/runs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from benchmarks.runner import execute_planned_run
from benchmarks.schema import PlannedRun, RunRecord, SuiteManifest

T = TypeVar("T")
R = TypeVar("R")


def effective_workers(parallel: int, n_tasks: int) -> int:
    """Clamp worker count to ``[1, n_tasks]`` (sequential when ``parallel <= 1``)."""
    if n_tasks <= 0:
        return 1
    if parallel <= 1:
        return 1
    return min(int(parallel), n_tasks)


def run_in_parallel(
    items: list[T],
    worker: Callable[[T], R],
    parallel: int,
) -> list[R]:
    """Run ``worker`` on each item; use a process pool when ``parallel > 1``."""
    if not items:
        return []
    workers = effective_workers(parallel, len(items))
    if workers == 1:
        return [worker(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, items))


@dataclass(frozen=True)
class TrainSeedJob:
    controller: str
    variant: str
    seed: int
    episodes: int | None
    checkpoint_dir: Path
    config_path: Path | None = None
    smoke: bool = False


def train_seed_worker(job: TrainSeedJob) -> dict[str, Any]:
    """Train one seed in an isolated process with its own MATLAB engine."""
    from dataclasses import replace

    ckpt_path = job.checkpoint_dir / f"{job.variant}_train{job.seed}.pt"

    if job.controller == "snn":
        from controllers.snn.config import SNNConfig
        from controllers.snn.trainer import train_dsqn

        config = SNNConfig(variant=job.variant, seed=int(job.seed), log_episodes=True)
        if job.smoke:
            config = config.for_smoke(
                episodes=int(job.episodes) if job.episodes is not None else 2,
                max_steps=10,
            )
        elif job.episodes is not None:
            config = replace(config, num_episodes=int(job.episodes))
        plan: dict[str, Any] = {
            "controller": job.controller,
            "variant": job.variant,
            "seed": job.seed,
            "episodes": config.num_episodes,
            "checkpoint": ckpt_path.as_posix(),
        }
        train_dsqn(config=config, checkpoint_path=ckpt_path)
        return {**plan, "status": "ok"}

    from controllers.ddpg.config import DDPGConfig
    from rl_adaptive_dbs.env_factory import build_mehregan_env

    config = DDPGConfig(variant=job.variant, seed=int(job.seed))
    if job.episodes is not None:
        config = replace(config, num_episodes=int(job.episodes))
    plan = {
        "controller": job.controller,
        "variant": job.variant,
        "seed": job.seed,
        "episodes": config.num_episodes,
        "checkpoint": ckpt_path.as_posix(),
    }

    env = build_mehregan_env(config_path=job.config_path)
    try:
        from controllers.ddpg import train

        train(env, config, checkpoint_path=ckpt_path)
        return {**plan, "status": "ok"}
    finally:
        env.close()


@dataclass(frozen=True)
class EvalSeedJob:
    controller: str
    variant: str
    seed: int
    checkpoint: Path | None
    suite: SuiteManifest
    write_timeseries: bool
    run_id: str | None
    config_path: Path | None = None


def eval_seed_worker(job: EvalSeedJob) -> RunRecord:
    """Evaluate one seed in an isolated process with its own MATLAB engine."""
    from benchmarks.schema import ControllerEntry, PlannedRun
    from rl_adaptive_dbs.env_factory import build_mehregan_env

    planned = PlannedRun(
        controller=job.controller,
        variant=job.variant,
        seed=int(job.seed),
        entry=ControllerEntry(
            controller=job.controller,
            variant=job.variant,
            checkpoint=job.checkpoint,
        ),
        checkpoint=job.checkpoint,
    )
    env = build_mehregan_env(config_path=job.config_path)
    try:
        return execute_planned_run(
            env,
            planned,
            job.suite,
            run_id=job.run_id,
            write_timeseries=job.write_timeseries,
        )
    finally:
        env.close()


@dataclass(frozen=True)
class BenchmarkRunJob:
    planned: PlannedRun
    suite: SuiteManifest
    write_timeseries: bool
    config_path: Path | None = None


def benchmark_run_worker(job: BenchmarkRunJob) -> RunRecord:
    """Execute one benchmark planned run in an isolated process."""
    from rl_adaptive_dbs.env_factory import build_mehregan_env

    env = build_mehregan_env(config_path=job.config_path)
    try:
        return execute_planned_run(
            env,
            job.planned,
            job.suite,
            write_timeseries=job.write_timeseries,
        )
    finally:
        env.close()
