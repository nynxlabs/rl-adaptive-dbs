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
from envs.mehregan import MehreganEnv
from rl_adaptive_dbs.info import CONTROLLER_VARIANTS


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
    if controller != "ddpg":
        msg = f"eval for {controller!r} is not implemented (Phase 5)"
        raise NotImplementedError(msg)


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
    default = repo_root / "artifacts" / "ddpg" / f"{variant}_train{train_seed}.pt"
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
) -> list[RunRecord]:
    validate_eval_request(controller, variant)
    repo_root = find_repo_root()
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

    env = MehreganEnv()
    snapshot = env_snapshot(env)
    records: list[RunRecord] = []
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

            if results_dir is not None:
                suite_dir = results_dir / suite_label
                record.run_dir = suite_dir / "runs" / run_dir_name(planned, record.run_id)
                write_run_outputs(suite_dir, record, write_timeseries=write_timeseries)
    finally:
        env.close()

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
