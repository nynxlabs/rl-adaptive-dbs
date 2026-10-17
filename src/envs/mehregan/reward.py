"""Mehregan reward Eq. (8)."""

from __future__ import annotations

import numpy as np

REWARD_BELOW_THRESHOLD = ("square", "linear")


def mehregan_reward(
    observation: np.ndarray,
    *,
    beta_threshold: float = 0.35,
    reward_scale: float = 10.0,
    below_threshold: str = "linear",
) -> float:
    """Instantaneous reward from normalized observation window (environment.md §6).

    Above $\\beta_t$ both readings are Eq. (8)'s quadratic penalty. Below it the
    printed Eq. (8) is ``(s_sum - beta_t) * 10``, which is negative there and so
    contradicts the prose ("a positive reward if it falls below the threshold").
    ``"linear"`` (default) is the printed branch with its sign flipped,
    $(\\beta_t - s_{sum}) \\cdot 10$. ``"square"`` is the curve the paper plots in
    Fig. 3c, $+((\\beta_t - s_{sum}) \\cdot 10)^2$ (digitized: $R(0) \\approx 11$,
    fitted exponent 2.1). Making ``"square"`` the default waits on the maintainer.
    """
    obs = np.asarray(observation, dtype=float).reshape(-1)
    if obs.size == 0:
        msg = "observation must be non-empty"
        raise ValueError(msg)
    if below_threshold not in REWARD_BELOW_THRESHOLD:
        msg = f"below_threshold must be one of {REWARD_BELOW_THRESHOLD}, got {below_threshold!r}"
        raise ValueError(msg)
    s_sum = float(np.mean(obs))
    delta = (s_sum - beta_threshold) * reward_scale
    if s_sum < beta_threshold:
        return delta**2 if below_threshold == "square" else -delta
    return -(delta**2)
