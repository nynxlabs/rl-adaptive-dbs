"""DDPG actor–critic (Mehregan et al.)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from controllers.ddpg.buffer import ReplayBuffer, Transition
from controllers.ddpg.checkpoint import load_actor, load_checkpoint, qat_state_dict_from_checkpoint, save_checkpoint
from controllers.ddpg.config import DDPGConfig, init_baseline_for_variant
from controllers.ddpg.eval import EvalConfig, RolloutResult, run_mehregan_eval, run_policy_rollout
from controllers.ddpg.networks import Actor, Critic, clone_module, hard_update, soft_update
from controllers.ddpg.quantization import QATActor, is_ptq_variant, prepare_actor_for_eval, unwrap_actor
from controllers.ddpg.replication import (
    ReplicationConfig,
    ReplicationResult,
    baseline_names_for_variant,
    run_replication,
    write_replication_summary,
)

from controllers.ddpg.trainer import DDPGTrainer, TrainMetrics, TrainResult, train_ddpg

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv

__all__ = [
    "Actor",
    "Critic",
    "DDPGConfig",
    "DDPGTrainer",
    "EvalConfig",
    "ReplicationConfig",
    "ReplicationResult",
    "ReplayBuffer",
    "RolloutResult",
    "TrainMetrics",
    "TrainResult",
    "Transition",
    "baseline_names_for_variant",
    "clone_module",
    "default_train_env",
    "evaluate",
    "hard_update",
    "init_baseline_for_variant",
    "load_actor",
    "load_checkpoint",
    "run_mehregan_eval",
    "run_policy_rollout",
    "run_replication",
    "save_checkpoint",
    "soft_update",
    "train",
    "train_ddpg",
    "write_replication_summary",
]


def default_train_env(config: DDPGConfig | None = None) -> MehreganEnv:
    """Build the default ``MehreganEnv`` for :func:`train` when ``env`` is omitted."""
    from envs.mehregan import MehreganEnv, MehreganEnvConfig

    cfg = (config or DDPGConfig()).with_variant_defaults()
    env_cfg = MehreganEnvConfig(
        max_episode_steps=cfg.max_episode_steps,
        state_length=1,
        action_space_mode=cfg.action_space_mode,  # type: ignore[arg-type]
        pattern_mean_hz=cfg.effective_pattern_mean_hz,
    )
    if cfg.action_space_mode == "fixed_mean_pattern":
        from envs.plant.python_backend import PythonPlant
        from rl_adaptive_dbs.user_config import resolve_config

        plant = PythonPlant(config=resolve_config().plant)
        return MehreganEnv(plant=plant, config=env_cfg)
    return MehreganEnv(config=env_cfg)


def train(
    env: MehreganEnv | None = None,
    config: DDPGConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = 50,
    **kwargs: Any,
) -> TrainResult:
    """Train DDPG on ``env`` (default: :func:`default_train_env`) and optionally save a checkpoint."""
    cfg = (config or DDPGConfig()).with_variant_defaults()
    if env is None:
        env = default_train_env(cfg)

    result = train_ddpg(
        env,
        cfg,
        resume_path=str(resume_path) if resume_path is not None else None,
        start_episode=start_episode,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_interval=checkpoint_interval,
        **kwargs,
    )
    return result


def evaluate(
    env: MehreganEnv,
    checkpoint: str | Path | Actor | nn.Module,
    *,
    config: EvalConfig | None = None,
    protocol: str = "mehregan_eval",
    variant: str | None = None,
) -> dict[str, Any]:
    """Evaluate a trained policy (checkpoint path or ``Actor``) on ``env``."""
    device = config.device if config is not None else "cpu"
    eval_variant: str
    qat_state: dict[str, torch.Tensor] | None = None

    if isinstance(checkpoint, Actor):
        actor = checkpoint
        eval_variant = variant or "paper"
    elif isinstance(checkpoint, nn.Module):
        actor = unwrap_actor(checkpoint)
        eval_variant = variant or "paper"
        if isinstance(checkpoint, QATActor):
            qat_state = checkpoint.state_dict()
    else:
        ckpt_payload = load_checkpoint(checkpoint, device="cpu")
        actor, ddpg_config = load_actor(checkpoint, device="cpu")
        eval_variant = variant or ddpg_config.variant
        qat_state = qat_state_dict_from_checkpoint(ckpt_payload)

    policy = prepare_actor_for_eval(
        actor,
        eval_variant,
        device=device,
        qat_state_dict=qat_state if eval_variant == "qat" else None,
    )
    eval_device = device if eval_variant != "ptq-int8" else "cpu"

    if protocol == "mehregan_eval":
        eval_payload = run_mehregan_eval(
            env,
            policy,
            config=(config or EvalConfig(device=eval_device)),
        )
        if is_ptq_variant(eval_variant):
            eval_payload.setdefault("metrics_extra", {})["quantization"] = eval_variant
        return eval_payload
    if protocol == "training_episode":
        rollout = run_policy_rollout(
            env,
            policy,
            seed=None if config is None else config.seed,
            device=eval_device,
        )
        return rollout.to_dict()
    msg = f"unknown eval protocol {protocol!r}"
    raise ValueError(msg)
