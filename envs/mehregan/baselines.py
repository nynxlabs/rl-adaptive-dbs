"""Fixed stimulation baselines (environment.md §4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from envs.mehregan.patterns import PatternAlphabet
from envs.plant.dbs import DbsSpec

if TYPE_CHECKING:
    from envs.mehregan.env import MehreganEnv


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    dbs_spec: DbsSpec
    description: str


def default_baselines() -> dict[str, BaselineSpec]:
    return {
        "none": BaselineSpec(
            "none",
            DbsSpec.none(),
            "No STN stimulation",
        ),
        "cdbs-130hz": BaselineSpec(
            "cdbs-130hz",
            DbsSpec.from_frequency_hz(130.0),
            "Conventional 130 Hz cDBS",
        ),
        "periodic-45hz": BaselineSpec(
            "periodic-45hz",
            DbsSpec.from_frequency_hz(45.0),
            "Periodic 45 Hz stimulation",
        ),
        "periodic-30hz": BaselineSpec(
            "periodic-30hz",
            DbsSpec.from_frequency_hz(30.0),
            "Periodic 30 Hz (init-30hz experiment)",
        ),
    }


def baseline_action(name: str, alphabet: PatternAlphabet | None = None) -> int:
    spec = default_baselines()[name]
    alpha = alphabet or PatternAlphabet()
    return alpha.action_for_dbs_spec(spec.dbs_spec)


def run_baseline_rollout(
    env: MehreganEnv,
    name: str,
    *,
    seed: int | None = None,
) -> dict[str, object]:
    """Run one episode with a fixed baseline action each step."""
    if name not in default_baselines():
        msg = f"unknown baseline {name!r}"
        raise ValueError(msg)
    action = baseline_action(name, env.alphabet)
    observation, info = env.reset(seed=seed)
    total_reward = float(info.get("reward", 0.0))
    rewards: list[float] = [total_reward]
    p_beta_trace: list[float] = [float(info.get("p_beta_raw", np.nan))]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        p_beta_trace.append(float(info.get("p_beta_raw", np.nan)))

    return {
        "baseline": name,
        "seed": seed,
        "action": action,
        "total_reward": total_reward,
        "rewards": rewards,
        "p_beta": p_beta_trace,
        "steps": len(rewards) - 1,
    }
