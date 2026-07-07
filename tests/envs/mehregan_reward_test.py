"""Mehregan reward Eq. (8)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.mehregan.reward import mehregan_reward


def test_reward_linear_branch_below_threshold() -> None:
    obs = np.array([0.30], dtype=float)
    # Below beta_t: positive reward (paper §III.C, Fig. 3c).
    assert mehregan_reward(obs) == pytest.approx((0.35 - 0.30) * 10.0)


def test_reward_quadratic_branch_at_or_above_threshold() -> None:
    obs = np.array([0.40], dtype=float)
    delta = (0.40 - 0.35) * 10.0
    assert mehregan_reward(obs) == pytest.approx(-(delta**2))


def test_reward_averages_observation_window() -> None:
    obs = np.array([0.30, 0.34], dtype=float)
    s_sum = 0.32
    assert mehregan_reward(obs) == pytest.approx((0.35 - s_sum) * 10.0)
