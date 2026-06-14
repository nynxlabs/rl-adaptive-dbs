"""Policy rollout and Mehregan eval protocol (§IV.A.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from controllers.ddpg.networks import Actor

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv


@dataclass(frozen=True)
class EvalConfig:
    """Post-training eval defaults aligned with [environment.md](../../docs/environment.md) §8."""

    seed: int = 0
    device: str = "cpu"
    # §IV.A.2: reset segment (via env.reset) + repeated policy steps.
    eval_steps: int = 5


@dataclass
class RolloutResult:
    """One rollout trace (training episode or eval segment)."""

    seed: int | None
    total_reward: float
    rewards: list[float] = field(default_factory=list)
    p_beta: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    stim_freq_hz: list[float] = field(default_factory=list)
    steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "total_reward": self.total_reward,
            "rewards": self.rewards,
            "p_beta": self.p_beta,
            "actions": self.actions,
            "stim_freq_hz": self.stim_freq_hz,
            "steps": self.steps,
            "p_beta_mean": self.p_beta_mean,
            "p_beta_final": self.p_beta_final,
            "stim_frequency_mean": self.stim_frequency_mean,
            "reward_sum": self.total_reward,
        }

    @property
    def p_beta_mean(self) -> float:
        if not self.p_beta:
            return float("nan")
        return float(np.mean(self.p_beta))

    @property
    def p_beta_final(self) -> float:
        if not self.p_beta:
            return float("nan")
        return float(self.p_beta[-1])

    @property
    def stim_frequency_mean(self) -> float:
        if not self.stim_freq_hz:
            return float("nan")
        return float(np.mean(self.stim_freq_hz))

    @property
    def reward_sum(self) -> float:
        return self.total_reward


def select_policy_action(actor: Actor, state: np.ndarray, *, device: str = "cpu") -> int:
    """Greedy action from actor logits (softmax + argmax)."""
    actor.eval()
    with torch.no_grad():
        state_t = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        logits = actor(state_t)
        action_t, _ = Actor.select_action(logits)
        return int(action_t.item())


def run_policy_rollout(
    env: MehreganEnv,
    actor: Actor,
    *,
    seed: int | None = None,
    device: str = "cpu",
) -> RolloutResult:
    """Run one full training episode with a fixed policy (same shape as ``run_baseline_rollout``)."""
    observation, info = env.reset(seed=seed)
    result = RolloutResult(
        seed=seed,
        total_reward=float(info.get("reward", 0.0)),
        rewards=[float(info.get("reward", 0.0))],
        p_beta=[float(info.get("p_beta_raw", np.nan))],
        stim_freq_hz=[float(info.get("dbs_freq_hz", 0.0))],
    )

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = select_policy_action(actor, observation, device=device)
        observation, reward, terminated, truncated, info = env.step(action)
        result.total_reward += reward
        result.rewards.append(reward)
        result.p_beta.append(float(info.get("p_beta_raw", np.nan)))
        result.actions.append(action)
        result.stim_freq_hz.append(float(info.get("dbs_freq_hz", 0.0)))

    result.steps = len(result.actions)
    return result


def run_mehregan_eval(
    env: MehreganEnv,
    actor: Actor,
    *,
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Mehregan §IV.A.2 eval: fixed seed, reset baseline, then ``eval_steps`` policy steps."""
    cfg = config or EvalConfig()
    observation, info = env.reset(seed=cfg.seed)
    result = RolloutResult(
        seed=cfg.seed,
        total_reward=float(info.get("reward", 0.0)),
        rewards=[float(info.get("reward", 0.0))],
        p_beta=[float(info.get("p_beta_raw", np.nan))],
        stim_freq_hz=[float(info.get("dbs_freq_hz", 0.0))],
    )

    for _ in range(cfg.eval_steps):
        action = select_policy_action(actor, observation, device=cfg.device)
        observation, reward, terminated, truncated, info = env.step(action)
        result.total_reward += reward
        result.rewards.append(reward)
        result.p_beta.append(float(info.get("p_beta_raw", np.nan)))
        result.actions.append(action)
        result.stim_freq_hz.append(float(info.get("dbs_freq_hz", 0.0)))
        if terminated or truncated:
            break

    result.steps = len(result.actions)
    payload = result.to_dict()
    payload["protocol"] = "mehregan_eval"
    payload["eval_steps"] = cfg.eval_steps
    return payload
