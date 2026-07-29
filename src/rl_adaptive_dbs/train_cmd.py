"""``rl-dbs train`` command implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.quantization import is_ptq_variant
from rl_adaptive_dbs.env_factory import build_mehregan_env
from rl_adaptive_dbs.info import CONTROLLER_VARIANTS


_CONTROLLER_PHASE: dict[str, int] = {}


def validate_train_request(controller: str, variant: str) -> None:
    if controller not in {"ddpg", "snn", "sea_dbs"}:
        msg = f"unknown controller {controller!r}; valid: ddpg, snn, sea_dbs"
        raise KeyError(msg)
    if variant not in CONTROLLER_VARIANTS[controller]:
        msg = f"unknown variant {variant!r} for controller {controller!r}"
        raise KeyError(msg)
    if controller in _CONTROLLER_PHASE:
        phase = _CONTROLLER_PHASE[controller]
        msg = f"training for {controller!r} is not implemented (Phase {phase})"
        raise NotImplementedError(msg)
    if controller == "ddpg" and is_ptq_variant(variant):
        msg = (
            f"variant {variant!r} is PTQ (post-training quantization, eval-only); "
            "train variant 'paper' first, then eval with "
            f"'rl-dbs eval --controller ddpg --variant {variant}' or "
            f"scripts/replicate_mehregan_ddpg.py --variant {variant}"
        )
        raise ValueError(msg)


def default_checkpoint_dir(controller: str, variant: str) -> Path:
    """Checkpoint root — files are ``{variant}_train{seed}.pt`` (flat, per suite YAML)."""
    del variant  # filename carries variant; dir is per-controller
    return Path("artifacts") / controller


def train_controller(
    controller: str,
    variant: str,
    *,
    seeds: tuple[int, ...],
    episodes: int | None = None,
    checkpoint_dir: Path | None = None,
    dry_run: bool = False,
    parallel: int = 1,
    config_path: Path | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    validate_train_request(controller, variant)
    out_dir = checkpoint_dir or default_checkpoint_dir(controller, variant)
    summaries: list[dict[str, Any]] = []

    for seed in seeds:
        if controller == "sea_dbs":
            from controllers.sea_dbs.config import SEADBSConfig

            config = SEADBSConfig(variant=variant, seed=int(seed), log_episodes=True)
            if smoke:
                config = config.for_smoke(
                    episodes=int(episodes) if episodes is not None else 2,
                    max_steps=5,
                )
            elif episodes is not None:
                config = replace(config, num_episodes=int(episodes))
            n_episodes = config.num_episodes
        elif controller == "snn":
            from controllers.snn.config import SNNConfig

            config = SNNConfig(variant=variant, seed=int(seed), log_episodes=True)
            if smoke:
                config = config.for_smoke(
                    episodes=int(episodes) if episodes is not None else 2,
                    max_steps=10,
                )
            elif episodes is not None:
                config = replace(config, num_episodes=int(episodes))
            n_episodes = config.num_episodes
        else:
            config = DDPGConfig(variant=variant, seed=int(seed))
            if episodes is not None:
                config = replace(config, num_episodes=int(episodes))
            n_episodes = config.num_episodes

        ckpt_path = out_dir / f"{variant}_train{seed}.pt"
        plan = {
            "controller": controller,
            "variant": variant,
            "seed": seed,
            "episodes": n_episodes,
            "checkpoint": ckpt_path.as_posix(),
        }
        if dry_run:
            summaries.append({**plan, "dry_run": True})
            continue

    if dry_run:
        return summaries

    if parallel > 1 and len(seeds) > 1:
        from rl_adaptive_dbs.parallel_workers import TrainSeedJob, run_in_parallel, train_seed_worker

        jobs = [
            TrainSeedJob(
                controller=controller,
                variant=variant,
                seed=int(seed),
                episodes=episodes,
                checkpoint_dir=out_dir,
                config_path=config_path,
                smoke=smoke,
            )
            for seed in seeds
        ]
        return run_in_parallel(jobs, train_seed_worker, parallel)

    for seed in seeds:
        ckpt_path = out_dir / f"{variant}_train{seed}.pt"
        if controller == "sea_dbs":
            from controllers.sea_dbs.config import SEADBSConfig
            from controllers.sea_dbs.trainer import train_sea_dbs

            config = SEADBSConfig(variant=variant, seed=int(seed), log_episodes=True)
            if smoke:
                config = config.for_smoke(
                    episodes=int(episodes) if episodes is not None else 2,
                    max_steps=5,
                )
            elif episodes is not None:
                config = replace(config, num_episodes=int(episodes))
            plan = {
                "controller": controller,
                "variant": variant,
                "seed": seed,
                "episodes": config.num_episodes,
                "checkpoint": ckpt_path.as_posix(),
                "metrics": ckpt_path.with_suffix(".metrics.json").as_posix(),
                "smoke": smoke,
            }
            train_sea_dbs(config=config, checkpoint_path=ckpt_path)
            summaries.append({**plan, "status": "ok"})
            continue

        if controller == "snn":
            from controllers.snn.config import SNNConfig
            from controllers.snn.trainer import train_dsqn

            config = SNNConfig(variant=variant, seed=int(seed), log_episodes=True)
            if smoke:
                config = config.for_smoke(
                    episodes=int(episodes) if episodes is not None else 2,
                    max_steps=10,
                )
            elif episodes is not None:
                config = replace(config, num_episodes=int(episodes))
            plan = {
                "controller": controller,
                "variant": variant,
                "seed": seed,
                "episodes": config.num_episodes,
                "checkpoint": ckpt_path.as_posix(),
                "metrics": ckpt_path.with_suffix(".metrics.json").as_posix(),
                "smoke": smoke,
            }
            train_dsqn(config=config, checkpoint_path=ckpt_path)
            summaries.append({**plan, "status": "ok"})
            continue

        config = DDPGConfig(variant=variant, seed=int(seed))
        if episodes is not None:
            config = replace(config, num_episodes=int(episodes))
        plan = {
            "controller": controller,
            "variant": variant,
            "seed": seed,
            "episodes": config.num_episodes,
            "checkpoint": ckpt_path.as_posix(),
        }

        env = build_mehregan_env(config_path=config_path)
        try:
            from controllers.ddpg import train

            train(env, config, checkpoint_path=ckpt_path)
            summaries.append({**plan, "status": "ok"})
        finally:
            env.close()
    return summaries
