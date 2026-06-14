"""DDPG actor–critic (Mehregan et al.)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from controllers.ddpg.buffer import ReplayBuffer, Transition
from controllers.ddpg.checkpoint import load_actor, load_checkpoint, save_checkpoint
from controllers.ddpg.config import DDPGConfig, init_baseline_for_variant
from controllers.ddpg.eval import EvalConfig, RolloutResult, run_mehregan_eval, run_policy_rollout
from controllers.ddpg.networks import Actor, Critic, clone_module, hard_update, soft_update
from controllers.ddpg.trainer import DDPGTrainer, TrainMetrics, TrainResult, train_ddpg

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv

__all__ = [
    "Actor",
    "Critic",
    "DDPGConfig",
    "DDPGTrainer",
    "EvalConfig",
    "ReplayBuffer",
    "RolloutResult",
    "TrainMetrics",
    "TrainResult",
    "Transition",
    "clone_module",
    "evaluate",
    "hard_update",
    "init_baseline_for_variant",
    "load_actor",
    "load_checkpoint",
    "run_mehregan_eval",
    "run_policy_rollout",
    "save_checkpoint",
    "soft_update",
    "train",
    "train_ddpg",
]


def train(
    env: MehreganEnv | None = None,
    config: DDPGConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    **kwargs: Any,
) -> TrainResult:
    """Train DDPG on ``env`` (default: ``MehreganEnv()``) and optionally save a checkpoint."""
    if env is None:
        from envs.mehregan import MehreganEnv

        env = MehreganEnv()

    result = train_ddpg(env, config, **kwargs)
    if checkpoint_path is not None:
        cfg = result.config
        save_checkpoint(
            checkpoint_path,
            actor=result.actor,
            config=cfg,
            state_length=int(env.observation_space.shape[0]),
            n_actions=int(env.action_space.n),
        )
    return result


def evaluate(
    env: MehreganEnv,
    checkpoint: str | Path | Actor,
    *,
    config: EvalConfig | None = None,
    protocol: str = "mehregan_eval",
) -> dict[str, Any]:
    """Evaluate a trained policy (checkpoint path or ``Actor``) on ``env``."""
    if isinstance(checkpoint, Actor):
        actor = checkpoint
        device = (config.device if config is not None else "cpu")
    else:
        device = config.device if config is not None else "cpu"
        actor, _ddpg_config = load_actor(checkpoint, device=device)

    if protocol == "mehregan_eval":
        return run_mehregan_eval(env, actor, config=config or EvalConfig(device=device))
    if protocol == "training_episode":
        rollout = run_policy_rollout(
            env,
            actor,
            seed=None if config is None else config.seed,
            device=device,
        )
        return rollout.to_dict()
    msg = f"unknown eval protocol {protocol!r}"
    raise ValueError(msg)
