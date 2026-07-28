"""Reward helpers for within-step Mehregan extensions."""

from __future__ import annotations

import numpy as np


def mehregan_reward_from_s_sum(
    s_sum: float,
    *,
    beta_threshold: float = 0.35,
    reward_scale: float = 10.0,
) -> float:
    """Instantaneous reward from scalar s_sum (environment.md section 6)."""
    delta = (float(s_sum) - beta_threshold) * reward_scale
    if s_sum < beta_threshold:
        return -delta
    return -(delta**2)


def mehregan_reward(
    observation: np.ndarray,
    *,
    beta_threshold: float = 0.35,
    reward_scale: float = 10.0,
) -> float:
    """Instantaneous reward from normalized observation window."""
    obs = np.asarray(observation, dtype=float).reshape(-1)
    if obs.size == 0:
        msg = "observation must be non-empty"
        raise ValueError(msg)
    return mehregan_reward_from_s_sum(
        float(np.mean(obs)),
        beta_threshold=beta_threshold,
        reward_scale=reward_scale,
    )
