"""Ravivarapu Eq. (7) reward from mean beta power."""

from __future__ import annotations

import numpy as np


def sea_dbs_reward(
    mean_p_beta: float,
    *,
    beta_threshold: float = 0.35,
    reward_scale: float = 10.0,
) -> float:
    """Instantaneous reward from average normalized beta (replication.md §6)."""
    delta = (float(mean_p_beta) - beta_threshold) * reward_scale
    if mean_p_beta < beta_threshold:
        return float(delta**2)
    return float(-(delta**2))
