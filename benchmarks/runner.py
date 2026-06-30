"""Execute benchmark suites and write ``results/`` outputs."""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.git import git_commit_short
from benchmarks.metrics import rollout_timeseries, rollout_to_core_metrics
from benchmarks.results import (
    build_suite_manifest_payload,
    suite_output_dir,
    utc_now_iso,
    write_run_outputs,
    write_suite_manifest,
)
from benchmarks.schema import PlannedRun, RunRecord, SuiteManifest
from benchmarks.suite import expand_planned_runs, find_repo_root, load_suite
from envs.mehregan.baselines import default_baselines, run_baseline_mehregan_eval
from envs.mehregan.env import MehreganEnv


@dataclass
class BenchmarkOptions:
    results_dir: Path = Path("results")
    seeds: tuple[int, ...] | None = None
    controller_filter: set[tuple[str, str]] | None = None
    dry_run: bool = False
    write_timeseries: bool = True
    repo_root: Path | None = None


@dataclass
class BenchmarkResult:
    suite: SuiteManifest
    suite_dir: Path
    planned: list[PlannedRun]
    records: list[RunRecord]
    manifest_path: Path


def make_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{stamp}-{suffix}"


def run_dir_name(planned: PlannedRun, run_id: str) -> str:
    return f"{planned.controller}_{planned.variant}_{run_id}"


def env_snapshot(env: MehreganEnv) -> dict[str, Any]:
    cfg = env.config
    return {
        "step_duration_s": cfg.step_duration_s,
        "max_episode_steps": cfg.max_episode_steps,
        "beta_threshold": cfg.beta_threshold,
        "reward_scale": cfg.reward_scale,
        "observation_scale": cfg.observation_scale,
        "state_length": cfg.state_length,
        "n_actions": int(env.action_space.n),
    }


def _execute_mehregan_run(
    env: MehreganEnv,
    planned: PlannedRun,
    suite: SuiteManifest,
) -> dict[str, Any]:
    if planned.controller == "baseline":
        if planned.variant not in default_baselines():
            msg = f"unknown baseline variant {planned.variant!r}"
            raise ValueError(msg)
        return run_baseline_mehregan_eval(
            env,
            planned.variant,
            seed=planned.seed,
            eval_steps=suite.eval_steps,
        )
    if planned.controller == "ddpg":
        from controllers.ddpg import EvalConfig, evaluate

        if planned.checkpoint is None or not planned.checkpoint.is_file():
            msg = f"checkpoint not found for ddpg/{planned.variant}: {planned.checkpoint}"
            raise FileNotFoundError(msg)
        return evaluate(
            env,
            planned.checkpoint,
            config=EvalConfig(seed=planned.seed, eval_steps=suite.eval_steps),
            protocol="mehregan_eval",
            variant=planned.variant,
        )
    msg = (
        f"controller {planned.controller!r} is not implemented for protocol "
        f"{suite.protocol!r} (Phase 5+)"
    )
    raise NotImplementedError(msg)


def execute_planned_run(
    env: MehreganEnv,
    planned: PlannedRun,
    suite: SuiteManifest,
    *,
    run_id: str | None = None,
    write_timeseries: bool = True,
) -> RunRecord:
    """Run one planned benchmark entry and build a ``RunRecord`` (no disk write)."""
    rid = run_id or make_run_id()
    repo_root = find_repo_root()
    suite_dir = suite_output_dir(Path("results"), suite)
    run_dir = suite_dir / "runs" / run_dir_name(planned, rid)

    if suite.protocol in {"mehregan", "mehregan_eval"}:
        payload = _execute_mehregan_run(env, planned, suite)
    else:
        msg = f"unsupported suite protocol {suite.protocol!r}"
        raise NotImplementedError(msg)

    metrics = rollout_to_core_metrics(
        payload,
        controller=planned.controller,
        variant=planned.variant,
        seed=planned.seed,
        run_id=rid,
        protocol=str(payload.get("protocol", suite.protocol)),
        step_duration_s=env.config.step_duration_s,
        metrics_extra=planned.entry.metrics_extra or None,
    )
    config: dict[str, Any] = {
        "controller": planned.controller,
        "variant": planned.variant,
        "seed": planned.seed,
        "run_id": rid,
        "suite": suite.name,
        "protocol": suite.protocol,
        "eval_steps": suite.eval_steps,
    }
    if planned.checkpoint is not None:
        config["checkpoint"] = str(planned.checkpoint)

    timeseries = rollout_timeseries(payload, step_duration_s=env.config.step_duration_s)
    record = RunRecord(
        planned=planned,
        run_id=rid,
        run_dir=run_dir,
        metrics=metrics,
        config=config,
        timeseries={"rollout": timeseries} if write_timeseries else None,
    )
    return record


def run_suite(
    suite: str | Path | SuiteManifest,
    env: MehreganEnv | None = None,
    *,
    options: BenchmarkOptions | None = None,
) -> BenchmarkResult:
    """Load (or accept) a suite manifest, execute all planned runs, write ``results/``."""
    opts = options or BenchmarkOptions()
    repo_root = opts.repo_root or find_repo_root()
    manifest = suite if isinstance(suite, SuiteManifest) else load_suite(suite, repo_root=repo_root)

    planned = expand_planned_runs(
        manifest,
        seeds=opts.seeds,
        controller_filter=opts.controller_filter,
        repo_root=repo_root,
    )

    suite_dir = suite_output_dir(opts.results_dir, manifest)
    started_at = utc_now_iso()
    git_commit = git_commit_short(repo_root)

    if opts.dry_run:
        manifest_path = write_suite_manifest(
            suite_dir,
            build_suite_manifest_payload(
                manifest,
                results_dir=opts.results_dir.resolve(),
                git_commit=git_commit,
                planned_runs=len(planned),
                completed_runs=0,
                started_at=started_at,
                finished_at=started_at,
                env_snapshot=None,
            ),
        )
        return BenchmarkResult(
            suite=manifest,
            suite_dir=suite_dir,
            planned=planned,
            records=[],
            manifest_path=manifest_path,
        )

    owns_env = env is None
    active_env = env
    if active_env is None:
        from rl_adaptive_dbs.env_factory import build_mehregan_env

        active_env = build_mehregan_env()

    records: list[RunRecord] = []
    try:
        snapshot = env_snapshot(active_env)
        for item in planned:
            record = execute_planned_run(
                active_env,
                item,
                manifest,
                write_timeseries=opts.write_timeseries,
            )
            record.run_dir = suite_dir / "runs" / run_dir_name(item, record.run_id)
            write_run_outputs(suite_dir, record, write_timeseries=opts.write_timeseries)
            records.append(record)

        manifest_path = write_suite_manifest(
            suite_dir,
            build_suite_manifest_payload(
                manifest,
                results_dir=opts.results_dir.resolve(),
                git_commit=git_commit,
                planned_runs=len(planned),
                completed_runs=len(records),
                started_at=started_at,
                finished_at=utc_now_iso(),
                env_snapshot=snapshot,
            ),
        )
    finally:
        if owns_env and active_env is not None:
            active_env.close()

    return BenchmarkResult(
        suite=manifest,
        suite_dir=suite_dir,
        planned=planned,
        records=records,
        manifest_path=manifest_path,
    )


def mehregan_env_config_snapshot() -> dict[str, Any]:
    """Default Mehregan env settings for manifest ``env`` block."""
    from rl_adaptive_dbs.user_config import resolve_config

    return asdict(resolve_config().env)
