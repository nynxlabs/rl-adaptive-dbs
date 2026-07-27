"""``rl-dbs eval`` command implementation."""

from __future__ import annotations

from pathlib import Path

from benchmarks.results import (
    build_suite_manifest_payload,
    utc_now_iso,
    write_run_outputs,
    write_suite_manifest,
)
from benchmarks.runner import env_snapshot, execute_planned_run, make_run_id, run_dir_name
from benchmarks.schema import ControllerEntry, PlannedRun, RunRecord, SuiteManifest
from benchmarks.suite import find_repo_root, load_suite
from rl_adaptive_dbs.env_factory import build_mehregan_env
from controllers.ddpg.quantization import fp_source_variant, is_ptq_variant
from rl_adaptive_dbs.info import CONTROLLER_VARIANTS


_CONTROLLER_PHASE: dict[str, int] = {"sea_dbs": 6}


def validate_eval_request(controller: str, variant: str) -> None:
    if controller == "baseline":
        if variant not in CONTROLLER_VARIANTS["baseline"]:
            msg = f"unknown baseline variant {variant!r}"
            raise KeyError(msg)
        return
    if controller not in {"ddpg", "snn", "sea_dbs"}:
        msg = f"unknown controller {controller!r}"
        raise KeyError(msg)
    if variant not in CONTROLLER_VARIANTS[controller]:
        msg = f"unknown variant {variant!r} for controller {controller!r}"
        raise KeyError(msg)
    if controller in _CONTROLLER_PHASE:
        phase = _CONTROLLER_PHASE[controller]
        msg = f"eval for {controller!r} is not implemented (Phase {phase})"
        raise NotImplementedError(msg)
    if controller == "snn":
        # Eval path for snn is wired below; keep sea_dbs blocked via _CONTROLLER_PHASE.
        return


def _resolve_checkpoint(
    controller: str,
    variant: str,
    checkpoint: Path | None,
    *,
    repo_root: Path,
    train_seed: int = 0,
) -> Path | None:
    if controller == "baseline":
        return None
    if checkpoint is not None:
        return checkpoint if checkpoint.is_absolute() else (repo_root / checkpoint).resolve()
    ckpt_variant = fp_source_variant(variant) if is_ptq_variant(variant) else variant
    default = repo_root / "artifacts" / controller / f"{ckpt_variant}_train{train_seed}.pt"
    return default


def _suite_manifest_for_eval(suite_name: str | None, *, repo_root: Path) -> SuiteManifest | None:
    if not suite_name:
        return None
    return load_suite(suite_name, repo_root=repo_root)


def eval_controller(
    controller: str,
    variant: str,
    *,
    seeds: tuple[int, ...],
    checkpoint: Path | None = None,
    suite_name: str | None = None,
    results_dir: Path | None = None,
    run_id: str | None = None,
    write_timeseries: bool = True,
    parallel: int = 1,
    config_path: Path | None = None,
    episodes: int | None = None,
    smoke: bool = False,
) -> list[RunRecord]:
    validate_eval_request(controller, variant)
    repo_root = find_repo_root()

    if controller == "snn":
        from controllers.snn.config import EVAL_EPISODES, EVAL_MAX_STEPS, SNNConfig
        from controllers.snn.eval import evaluate as evaluate_snn

        records: list[RunRecord] = []
        for seed in seeds:
            ckpt = _resolve_checkpoint(
                controller, variant, checkpoint, repo_root=repo_root, train_seed=int(seed)
            )
            if ckpt is None or not ckpt.is_file():
                msg = f"checkpoint not found: {ckpt}"
                raise FileNotFoundError(msg)
            cfg = SNNConfig(variant=variant, seed=int(seed))
            if smoke:
                cfg = cfg.for_smoke(episodes=int(episodes) if episodes is not None else 2, max_steps=10)
            eval_episodes = int(episodes) if episodes is not None else EVAL_EPISODES
            eval_steps = 10 if smoke else EVAL_MAX_STEPS
            payload = evaluate_snn(
                ckpt,
                config=cfg,
                episodes=eval_episodes,
                max_steps=eval_steps,
            )
            rid = run_id or make_run_id()
            planned = PlannedRun(
                controller=controller,
                variant=variant,
                seed=int(seed),
                entry=ControllerEntry(
                    controller=controller, variant=variant, checkpoint=ckpt, adapter=True
                ),
                checkpoint=ckpt,
            )
            records.append(
                RunRecord(
                    planned=planned,
                    run_id=rid,
                    run_dir=Path("results") / "adhoc_eval" / "runs" / run_dir_name(planned, rid),
                    metrics={
                        "controller": controller,
                        "variant": variant,
                        "seed": int(seed),
                        "run_id": rid,
                        "protocol": "nguyen_eval",
                        "p_beta_mean": payload["p_beta_mean"],
                        "reward_sum": payload["reward_sum"],
                        "alpha_beta_mean": payload["alpha_beta_mean"],
                        "n_episodes": payload["n_episodes"],
                    },
                    config={
                        "controller": controller,
                        "variant": variant,
                        "seed": int(seed),
                        "checkpoint": ckpt.as_posix(),
                        "protocol": "nguyen_eval",
                    },
                    timeseries=None,
                )
            )
        return records

    suite = _suite_manifest_for_eval(suite_name, repo_root=repo_root)
    protocol = suite.protocol if suite else "mehregan"
    eval_steps = suite.eval_steps if suite else 5
    suite_label = suite.name if suite else "adhoc_eval"

    if suite is None:
        suite = SuiteManifest(
            name=suite_label,
            version=1,
            protocol=protocol,
            seeds=seeds,
            controllers=(ControllerEntry(controller=controller, variant=variant),),
            eval_steps=eval_steps,
        )

    if parallel > 1 and len(seeds) > 1:
        from rl_adaptive_dbs.parallel_workers import EvalSeedJob, eval_seed_worker, run_in_parallel

        jobs: list[EvalSeedJob] = []
        for seed in seeds:
            ckpt = _resolve_checkpoint(controller, variant, checkpoint, repo_root=repo_root)
            if controller == "ddpg" and (ckpt is None or not ckpt.is_file()):
                msg = f"checkpoint not found: {ckpt}"
                raise FileNotFoundError(msg)
            jobs.append(
                EvalSeedJob(
                    controller=controller,
                    variant=variant,
                    seed=int(seed),
                    checkpoint=ckpt,
                    suite=suite,
                    write_timeseries=write_timeseries,
                    run_id=run_id,
                    config_path=config_path,
                )
            )
        records = run_in_parallel(jobs, eval_seed_worker, parallel)
        env = build_mehregan_env(config_path=config_path)
        try:
            snapshot = env_snapshot(env)
        finally:
            env.close()
    else:
        env = build_mehregan_env(config_path=config_path)
        snapshot = env_snapshot(env)
        records = []
        try:
            for seed in seeds:
                ckpt = _resolve_checkpoint(controller, variant, checkpoint, repo_root=repo_root)
                if controller == "ddpg" and (ckpt is None or not ckpt.is_file()):
                    msg = f"checkpoint not found: {ckpt}"
                    raise FileNotFoundError(msg)

                planned = PlannedRun(
                    controller=controller,
                    variant=variant,
                    seed=int(seed),
                    entry=ControllerEntry(controller=controller, variant=variant, checkpoint=ckpt),
                    checkpoint=ckpt,
                )
                rid = run_id or make_run_id()
                record = execute_planned_run(
                    env,
                    planned,
                    suite,
                    run_id=rid,
                    write_timeseries=write_timeseries,
                )
                records.append(record)
        finally:
            env.close()

    if results_dir is not None:
        suite_dir = results_dir / suite_label
        for seed, record in zip(seeds, records, strict=True):
            planned = PlannedRun(
                controller=controller,
                variant=variant,
                seed=int(seed),
                entry=ControllerEntry(controller=controller, variant=variant),
            )
            record.run_dir = suite_dir / "runs" / run_dir_name(planned, record.run_id)
            write_run_outputs(suite_dir, record, write_timeseries=write_timeseries)

    if results_dir is not None and records:
        suite_dir = results_dir / suite_label
        write_suite_manifest(
            suite_dir,
            build_suite_manifest_payload(
                suite,
                results_dir=results_dir.resolve(),
                git_commit=None,
                planned_runs=len(records),
                completed_runs=len(records),
                started_at=utc_now_iso(),
                finished_at=utc_now_iso(),
                env_snapshot=snapshot,
            ),
        )
    return records


def records_to_summary_lines(records: list[RunRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        m = record.metrics
        lines.append(
            f"{m['controller']}:{m['variant']} seed={m['seed']} "
            f"p_beta_mean={m['p_beta_mean']:.2f} reward_sum={m['reward_sum']:.2f}"
        )
    return lines
