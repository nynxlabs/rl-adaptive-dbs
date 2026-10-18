"""Fig 3a per-pulse logits design and the Alg. 1 agent used by the Mehregan candidate runs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from controllers.ddpg.action_designs import (
    AlgOneAgent,
    PerPulseLogits,
    StateCNN,
    fanin_param_groups,
    regular_onsets,
)


def test_state_cnn_fan_in_follows_state_length() -> None:
    assert StateCNN(state_len=2000).fc1.in_features == 32_000  # Figure 3a: flatten 1 x l*16
    assert StateCNN(state_len=1).fc1.in_features == 64
    assert StateCNN(state_len=1)(torch.zeros(3, 1)).shape == (3, 256)


def test_regular_init_plays_the_regular_train() -> None:
    design = PerPulseLogits(n_pulses=90, init_bias=1.0)
    agent = AlgOneAgent(design, state_len=1, actor_lr=5e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, seed=0)
    raw = agent.actor(torch.full((1, 1), 0.45))
    assert raw.shape == (1, 90 * 2000)
    np.testing.assert_array_equal(design.decode_raw(raw), regular_onsets(90))
    assert design.regular_margin(raw) > 0.5


def test_centered_view_removes_the_init_spike() -> None:
    design = PerPulseLogits(n_pulses=4, n_slots=10, init_bias=2.0, center=True)
    stored = torch.zeros(1, 40)
    stored.view(4, 10)[torch.arange(4), torch.as_tensor(regular_onsets(4, 10))] = 2.0
    assert torch.count_nonzero(design.view_stored(stored)) == 0
    assert torch.equal(PerPulseLogits(n_pulses=4, n_slots=10).view_stored(stored), stored)


def test_fanin_param_groups_scale_wide_layers_only() -> None:
    net = torch.nn.Sequential(torch.nn.Linear(32_000, 256), torch.nn.Linear(256, 10))
    lrs = [g["lr"] for g in fanin_param_groups(net, 5e-4, 256)]
    assert lrs == pytest.approx([5e-4 * 256 / 32_000, 5e-4])
    assert [g["lr"] for g in fanin_param_groups(net, 5e-4, 0)] == [5e-4]


def test_update_runs_and_moves_the_actor() -> None:
    design = PerPulseLogits(n_pulses=4, n_slots=20)
    agent = AlgOneAgent(design, state_len=1, actor_lr=5e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, seed=0)
    before = agent.actor.head.weight.detach().clone()
    s = torch.full((8, 1), 0.45)
    a = design.explore(agent.actor(s).detach(), 1.0, agent.gen)
    out = agent.update(s, a, -torch.ones(8), s, torch.zeros(8))
    assert np.isfinite(out["loss_c"]) and np.isfinite(out["q"])
    assert not torch.equal(before, agent.actor.head.weight)
