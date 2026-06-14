"""Mehregan DDPG paper replication workflow (train → eval → baselines)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.eval import EvalConfig
from controllers.ddpg.trainer import TrainMetrics, TrainResult
from envs.mehregan.baselines import run_baseline_rollout

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv


def baseline_names_for_variant(variant: str) -> tuple[str, ...]:
    """Baselines to compare against for a benchmark variant slug."""
    if variant == "init-30hz":
        return ("none", "cdbs-130hz", "periodic-30hz")
    return ("none", "cdbs-130hz", "periodic-45hz")


def _rollout_metrics(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    p_beta = [float(x) for x in payload["p_beta"]]
    stim = payload.get("stim_freq_hz")
    stim_mean = float(np.mean(stim)) if stim else float("nan")
    return {
        "label": label,
        "reward_sum": float(payload["total_reward"] if "total_reward" in payload else payload["reward_sum"]),
        "p_beta_mean": float(np.mean(p_beta)),
        "p_beta_final": float(p_beta[-1]),
        "stim_frequency_mean": stim_mean,
        "steps": int(payload.get("steps", len(payload.get("actions", [])))),
    }


@dataclass(frozen=True)
class ReplicationConfig:
    """One Mehregan §IV replication run (full-precision variants)."""

    variant: str = "paper"
    train_seed: int = 0
    eval_seed: int = 0
    ddpg: DDPGConfig | None = None
    eval: EvalConfig | None = None
    baselines: tuple[str, ...] | None = None
    checkpoint_dir: Path | None = None


@dataclass
class ReplicationResult:
    variant: str
    train_seed: int
    eval_seed: int
    train: TrainResult
    eval_metrics: dict[str, Any]
    baseline_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint_path: Path | None = None

    def summary(self) -> dict[str, Any]:
        """Compact comparison table for logging or JSON export."""
        return {
            "variant": self.variant,
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "train_episode_rewards": list(self.train.metrics.episode_rewards),
            "ddpg_eval": self.eval_metrics,
            "baselines": self.baseline_metrics,
        }


def _resolve_ddpg_config(config: ReplicationConfig) -> DDPGConfig:
    if config.ddpg is not None:
        return config.ddpg
    return DDPGConfig(variant=config.variant, seed=config.train_seed)


def _resolve_eval_config(config: ReplicationConfig, ddpg: DDPGConfig) -> EvalConfig:
    if config.eval is not None:
        return config.eval
    return EvalConfig(seed=config.eval_seed, device=ddpg.device)


def _checkpoint_path(config: ReplicationConfig, ddpg: DDPGConfig) -> Path | None:
    if config.checkpoint_dir is None:
        return None
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return config.checkpoint_dir / f"{ddpg.variant}_train{config.train_seed}.pt"


def run_replication(
    env: MehreganEnv,
    config: ReplicationConfig,
    *,
    checkpoint_path: Path | None = None,
) -> ReplicationResult:
    """Train DDPG, run ``mehregan_eval``, and compare to fixed baselines on ``env``."""
    from controllers.ddpg import evaluate, train

    ddpg = _resolve_ddpg_config(config)
    eval_cfg = _resolve_eval_config(config, ddpg)
    ckpt = checkpoint_path if checkpoint_path is not None else _checkpoint_path(config, ddpg)

    train_result = train(env, ddpg, checkpoint_path=ckpt)
    if ckpt is None:
        eval_target = train_result.actor
    else:
        eval_target = ckpt

    eval_metrics = evaluate(env, eval_target, config=eval_cfg, protocol="mehregan_eval")

    baseline_names = config.baselines or baseline_names_for_variant(config.variant)
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for name in baseline_names:
        rollout = run_baseline_rollout(env, name, seed=config.eval_seed)
        baseline_metrics[name] = _rollout_metrics(rollout, label=name)

    return ReplicationResult(
        variant=ddpg.variant,
        train_seed=config.train_seed,
        eval_seed=config.eval_seed,
        train=train_result,
        eval_metrics=eval_metrics,
        baseline_metrics=baseline_metrics,
        checkpoint_path=ckpt,
    )


def write_replication_summary(result: ReplicationResult, path: str | Path) -> Path:
    """Write ``ReplicationResult.summary()`` as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.summary()
    # TrainMetrics / nested objects are already plain lists and dicts in summary().
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def train_metrics_to_dict(metrics: TrainMetrics) -> dict[str, Any]:
    return asdict(metrics)
