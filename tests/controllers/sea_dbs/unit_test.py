"""Unit tests for SEA-DBS reward, GS, and variant flags."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.networks import gumbel_softmax_sample
from controllers.sea_dbs.reward import sea_dbs_reward
from controllers.sea_dbs.trainer import SEA_DBSTrainer


def test_adapter_reset_nonzero_p_beta() -> None:
    env = SEA_DBSEnvAdapter(config=SEADBSConfig(seed=0))
    try:
        _obs, info = env.reset(seed=0)
        assert info["p_beta_raw"] > 0.0
        assert info["mean_p_beta"] > 0.0
        obs, _reward, _term, _trunc, step_info = env.step(1)
        assert step_info["p_beta_raw"] > 0.0
        assert float(obs[0]) > 0.0
    finally:
        env.close()


def test_set_carrier_hz_survives_reset() -> None:
    """Fig 5 eval sets 50/30 Hz then reset(); must not restore train carrier."""
    from types import SimpleNamespace

    class _FakePlant:
        def reset(self, seed: int | None = None) -> None:
            return None

        def integrate(self, duration_s, dbs_spec=None, *, record_spikes: bool = True):
            return SimpleNamespace(p_beta=196.0)

        def close(self) -> None:
            return None

    cfg = SEADBSConfig(seed=0, carrier_hz=130.0, dbs_burst_ms=62.0)
    env = SEA_DBSEnvAdapter(plant=_FakePlant(), config=cfg)
    try:
        env.set_carrier_hz(50.0)
        _obs, info = env.reset(seed=0)
        assert info["carrier_hz"] == 50.0
        _obs, _reward, _term, _trunc, step_info = env.step(1)
        assert step_info["carrier_hz"] == 50.0
        env.set_carrier_hz(30.0)
        _obs, info = env.reset(seed=1)
        assert info["carrier_hz"] == 30.0
    finally:
        env.close()


def test_untreated_window_used_for_reset_and_skip() -> None:
    from types import SimpleNamespace

    durations: list[float] = []

    class _FakePlant:
        def reset(self, seed: int | None = None) -> None:
            return None

        def integrate(self, duration_s, dbs_spec=None, *, record_spikes: bool = True):
            durations.append(float(duration_s))
            return SimpleNamespace(p_beta=196.0)

        def close(self) -> None:
            return None

    cfg = replace(
        SEADBSConfig(seed=0, biomarker_window_s=0.15),
        untreated_window_s=0.10,
    )
    env = SEA_DBSEnvAdapter(plant=_FakePlant(), config=cfg)
    try:
        env.reset(seed=0)
        env.step(1)
        env.step(0)
    finally:
        env.close()
    assert durations[0] == pytest.approx(0.10)
    assert durations[1] == pytest.approx(0.15)
    assert durations[2] == pytest.approx(0.10)


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


def test_hard_gumbel_action_independent_of_tau() -> None:
    logits = torch.tensor([0.2, 0.5])
    torch.manual_seed(0)
    _, a_lo = gumbel_softmax_sample(logits, tau=0.42, hard=True)
    torch.manual_seed(0)
    _, a_hi = gumbel_softmax_sample(logits, tau=5.0, hard=True)
    assert int(a_lo.item()) == int(a_hi.item())


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


def test_gs_episode_schedule() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        gs_tau0=1.0,
        gs_tau_min=0.05,
        gs_lambda=1e-4,
        gs_early_lambda_episode_hi=3,
        gs_early_lambda_scale=10.0,
        gs_late_tau_floor_episode_lo=4,
        gs_late_tau_floor=0.5,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        trainer._total_steps = 1000
        trainer._current_episode = 1
        tau_boosted = trainer.gs_temperature()
        trainer._current_episode = 4
        tau_unboosted = trainer.gs_temperature()
        trainer._current_episode = 5
        tau_late_floor = trainer.gs_temperature()
        assert tau_boosted < tau_unboosted
        assert tau_late_floor >= 0.5
    finally:
        env.close()


def test_episode_stim_logit_boost() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        actor_mid_episode_lo=2,
        actor_mid_episode_hi=4,
        actor_mid_episode_stim_logit_boost=0.5,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        base = torch.tensor([1.0, -1.0])
        trainer._current_episode = 1
        assert torch.allclose(trainer._episode_action_logits(base), base)
        trainer._current_episode = 3
        boosted = trainer._episode_action_logits(base)
        assert float(boosted[0]) == pytest.approx(0.5)
        assert float(boosted[1]) == pytest.approx(-0.5)
    finally:
        env.close()


def test_episode_midlate_stim_logit_boost() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        actor_midlate_episode_lo=2,
        actor_midlate_episode_hi=4,
        actor_midlate_episode_stim_logit_boost=0.25,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        base = torch.tensor([1.0, -1.0])
        trainer._current_episode = 3
        boosted = trainer._episode_action_logits(base)
        assert float(boosted[0]) == pytest.approx(0.75)
        assert float(boosted[1]) == pytest.approx(-0.75)
    finally:
        env.close()


def test_episode_gap_patch_no_stim_boost() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        actor_gap_patch_episode_lo=2,
        actor_gap_patch_episode_hi=4,
        actor_gap_patch_no_stim_boost=0.15,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        base = torch.tensor([0.0, 0.0])
        trainer._current_episode = 3
        patched = trainer._episode_action_logits(base)
        assert float(patched[0]) == pytest.approx(0.15)
    finally:
        env.close()


def test_episode_late_no_stim_flat_boost() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        actor_late_episode_lo=4,
        actor_late_episode_hi=6,
        actor_late_episode_no_stim_boost=0.3,
        actor_late_episode_boost_ramp=False,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        base = torch.tensor([0.0, 0.0])
        trainer._current_episode = 4
        flat = trainer._episode_action_logits(base)
        assert float(flat[0]) == pytest.approx(0.3)
    finally:
        env.close()


def test_episode_late_no_stim_logit_boost() -> None:
    cfg = replace(
        SEADBSConfig(variant="baseline-gs").for_smoke(episodes=6, max_steps=2),
        actor_late_episode_lo=4,
        actor_late_episode_hi=6,
        actor_late_episode_no_stim_boost=0.3,
        actor_late_episode_boost_ramp=True,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        base = torch.tensor([0.0, 0.0])
        trainer._current_episode = 4
        ramp_start = trainer._episode_action_logits(base)
        assert float(ramp_start[0]) == pytest.approx(0.0)
        trainer._current_episode = 5
        late = trainer._episode_action_logits(base)
        assert float(late[0]) == pytest.approx(0.3)
        assert float(late[1]) == pytest.approx(-0.3)
    finally:
        env.close()


def test_ptq_weight_noise_is_deterministic_and_changes_params() -> None:
    from controllers.sea_dbs.networks import Actor
    from controllers.sea_dbs.quantization import apply_fp16_ptq, perturb_float_parameters

    actor = Actor(state_dim=1, n_actions=2, hidden_size=8)
    torch.manual_seed(0)
    with torch.no_grad():
        for param in actor.parameters():
            param.normal_(0.0, 0.1)
    a = perturb_float_parameters(actor, scale=0.03, seed=11)
    b = perturb_float_parameters(actor, scale=0.03, seed=11)
    c = perturb_float_parameters(actor, scale=0.03, seed=22)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa, pb)
    changed = False
    for pa, pc in zip(a.parameters(), c.parameters(), strict=True):
        if not torch.allclose(pa, pc):
            changed = True
            break
    assert changed
    fp16 = apply_fp16_ptq(actor, weight_noise=0.03, noise_seed=11)
    assert next(fp16.parameters()).dtype == torch.float16
