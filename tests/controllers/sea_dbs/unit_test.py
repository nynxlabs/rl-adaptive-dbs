"""Unit tests for SEA-DBS reward, GS, and variant flags."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.networks import gumbel_softmax_sample
from controllers.sea_dbs.reward import sea_dbs_reward


def test_reward_eq7_below_threshold_positive() -> None:
    r = sea_dbs_reward(0.2, beta_threshold=0.35, reward_scale=10.0)
    assert r > 0
    expected = ((0.2 - 0.35) * 10.0) ** 2
    assert r == pytest.approx(expected)


def test_reward_eq7_above_threshold_negative() -> None:
    r = sea_dbs_reward(0.5, beta_threshold=0.35, reward_scale=10.0)
    assert r < 0
    expected = -((0.5 - 0.35) * 10.0) ** 2
    assert r == pytest.approx(expected)


def test_gumbel_softmax_binary_shape() -> None:
    logits = torch.tensor([0.1, -0.2])
    relaxed, action = gumbel_softmax_sample(logits, tau=1.0, hard=True)
    assert relaxed.shape == (1, 2)
    assert int(action.item()) in {0, 1}


@pytest.mark.parametrize(
    ("variant", "pm", "gs"),
    [
        ("baseline", False, False),
        ("baseline-pm", True, False),
        ("baseline-gs", False, True),
        ("paper", True, True),
    ],
)
def test_variant_flags(variant: str, pm: bool, gs: bool) -> None:
    cfg = SEADBSConfig(variant=variant).with_variant_defaults()
    assert cfg.use_predictive_model is pm
    assert cfg.use_gumbel_softmax is gs


def test_for_smoke_reduces_episodes() -> None:
    cfg = SEADBSConfig().for_smoke(episodes=2, max_steps=4)
    assert cfg.num_episodes == 2
    assert cfg.max_episode_steps == 4
