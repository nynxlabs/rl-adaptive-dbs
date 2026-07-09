"""Mehregan reward Eq. (8)."""

from __future__ import annotations

import numpy as np


def mehregan_reward(
    observation: np.ndarray,
    *,
    beta_threshold: float = 0.35,
    reward_scale: float = 10.0,
) -> float:
    """Instantaneous reward from normalized observation window (environment.md §6)."""
    obs = np.asarray(observation, dtype=float).reshape(-1)
    if obs.size == 0:
        msg = "observation must be non-empty"
        raise ValueError(msg)
    s_sum = float(np.mean(obs))
    delta = (s_sum - beta_threshold) * reward_scale
    if s_sum < beta_threshold:
        # Paper §III.C / Fig. 3c: positive reward below beta_t. Eq. (8) as printed
        # yields negative delta here; negate to match prose and training intent.
        return -delta
    return -(delta**2)
