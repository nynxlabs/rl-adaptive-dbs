"""``rl-dbs train`` command implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from controllers.ddpg.config import DDPGConfig
from envs.mehregan import MehreganEnv
from rl_adaptive_dbs.info import CONTROLLER_VARIANTS


def validate_train_request(controller: str, variant: str) -> None:
    if controller not in {"ddpg", "snn", "sea_dbs"}:
        msg = f"unknown controller {controller!r}; valid: ddpg, snn, sea_dbs"
        raise KeyError(msg)
    if variant not in CONTROLLER_VARIANTS[controller]:
        msg = f"unknown variant {variant!r} for controller {controller!r}"
        raise KeyError(msg)
    if controller != "ddpg":
        msg = f"training for {controller!r} is not implemented (Phase 5)"
        raise NotImplementedError(msg)


def default_checkpoint_dir(controller: str, variant: str) -> Path:
    return Path("artifacts") / controller / variant


def train_controller(
    controller: str,
    variant: str,
    *,
    seeds: tuple[int, ...],
    episodes: int | None = None,
    checkpoint_dir: Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    validate_train_request(controller, variant)
    out_dir = checkpoint_dir or default_checkpoint_dir(controller, variant)
    summaries: list[dict[str, Any]] = []

    for seed in seeds:
        config = DDPGConfig(variant=variant, seed=int(seed))
        if episodes is not None:
            config = replace(config, num_episodes=int(episodes))
        ckpt_path = out_dir / f"{variant}_train{seed}.pt"
        plan = {
            "controller": controller,
            "variant": variant,
            "seed": seed,
            "episodes": config.num_episodes,
            "checkpoint": str(ckpt_path),
        }
        if dry_run:
            summaries.append({**plan, "dry_run": True})
            continue

        env = MehreganEnv()
        try:
            from controllers.ddpg import train

            train(env, config, checkpoint_path=ckpt_path)
            summaries.append({**plan, "status": "ok"})
        finally:
            env.close()
    return summaries
