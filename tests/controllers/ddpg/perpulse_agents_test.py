"""Deviation candidates for the per-pulse head (Mehregan Fig 4a study)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from controllers.ddpg.perpulse_agents import AgentKnobs, LogitsDesign, PulseGrid, make_agent


def _agent(design: str, algo: str = "ddpg", **kw):
    k = AgentKnobs(state_len=1, seed=0, sigma=1.0)
    opts = {"init_bias": 1.0, "noise": "gauss", "view": "raw", "delta_max": 3.0, "mix_init": -3.0} | kw
    agent, grid = make_agent(design=design, algo=algo, n_pulses=6, k=k, **opts)
    return agent, grid


@pytest.mark.parametrize("design", ["logits", "residual", "offset"])
def test_noise_free_init_is_the_regular_train(design: str) -> None:
    agent, grid = _agent(design)
    np.testing.assert_array_equal(agent.noise_free(torch.full((1, 1), 0.45)), grid.reg)


def test_mixture_starts_mostly_random() -> None:
    agent, _ = _agent("mixture")
    assert agent.diag(torch.full((1, 1), 0.45))["w"] == pytest.approx(0.047, abs=0.01)


def test_residual_logits_stay_within_delta_of_the_init() -> None:
    grid = PulseGrid(6)
    d = LogitsDesign(grid, init_bias=1.0, residual=True, delta_max=2.0)
    lg = d.logits(torch.full((1, d.raw_dim), 50.0))
    assert float((lg - grid.spike(1.0)).abs().max()) <= 2.0 + 1e-6


@pytest.mark.parametrize("view", ["regprob", "offhist"])
def test_compressed_views_of_played_and_expected_actions_agree_in_shape(view: str) -> None:
    grid = PulseGrid(6)
    d = LogitsDesign(grid, view=view, noise="gumbel")
    raw = torch.zeros(2, d.raw_dim)
    stored = d.explore(raw, 1.0, torch.Generator().manual_seed(0))
    assert d.view_stored(stored, 1.0).shape == d.view_actor(raw, 1.0).shape == (2, d.view_dim)
    if view == "offhist":  # histograms over pulses sum to one
        assert torch.allclose(d.view_actor(raw, 1.0).sum(-1), torch.ones(2), atol=1e-4)


def test_compressed_view_needs_gumbel() -> None:
    with pytest.raises(ValueError, match="gumbel"):
        LogitsDesign(PulseGrid(6), view="regprob", noise="gauss")


def test_reinforce_raises_the_probability_of_a_rewarded_pattern() -> None:
    agent, grid = _agent("logits", "reinforce", noise="gumbel")
    s = torch.full((4, 1), 0.45)
    target = torch.zeros(4, grid.n_pulses * grid.n_slots)
    target.view(4, grid.n_pulses, grid.n_slots)[:, torch.arange(grid.n_pulses), torch.as_tensor(grid.reg)] = 5.0
    other = torch.zeros_like(target)
    batch_a = torch.cat([target[:2], other[:2]])  # first two played the regular train
    r = torch.tensor([1.0, 1.0, -1.0, -1.0])
    agent.baseline = 0.0

    def regular_logp() -> float:
        with torch.no_grad():
            lg = agent.d._grid3(agent.d.logits(agent.actor(s[:1])))[0]
            return float(torch.log_softmax(lg, -1)[torch.arange(grid.n_pulses), torch.as_tensor(grid.reg)].sum())

    before = regular_logp()
    for _ in range(5):
        agent.update(s, batch_a, r, s, torch.zeros(4))
    assert regular_logp() > before


def test_td3_update_runs() -> None:
    agent, _ = _agent("offset", "td3")
    s = torch.full((8, 1), 0.45)
    a = torch.zeros(8, 6)
    out = agent.update(s, a, -torch.ones(8), s, torch.zeros(8))
    assert len(agent.critics) == 2 and np.isfinite(out["loss_c"])
