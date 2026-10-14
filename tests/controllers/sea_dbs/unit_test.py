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


def test_same_plant_window_for_stim_and_no_stim() -> None:
    """One P_beta estimator: reset, stim, and no-stim shots share the window."""
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

    env = SEA_DBSEnvAdapter(plant=_FakePlant(), config=SEADBSConfig(seed=0, biomarker_window_s=0.15))
    try:
        env.reset(seed=0)
        env.step(1)
        env.step(0)
    finally:
        env.close()
    assert durations == pytest.approx([0.15, 0.15, 0.15])


def test_prefilled_window_starts_at_untreated_mean() -> None:
    from types import SimpleNamespace

    values = iter([200.0, 100.0])

    class _FakePlant:
        def reset(self, seed: int | None = None) -> None:
            return None

        def integrate(self, duration_s, dbs_spec=None, *, record_spikes: bool = True):
            return SimpleNamespace(p_beta=next(values))

        def close(self) -> None:
            return None

    cfg = SEADBSConfig(seed=0, observation_scale=400.0, n_obs=5, prefill_obs_window=True)
    env = SEA_DBSEnvAdapter(plant=_FakePlant(), config=cfg)
    try:
        obs, _info = env.reset(seed=0)
        assert float(obs[0]) == pytest.approx(0.5)
        obs, *_ = env.step(1)
        assert float(obs[0]) == pytest.approx((4 * 0.5 + 0.25) / 5)
    finally:
        env.close()


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


def test_gs_temperature_follows_eq14() -> None:
    cfg = SEADBSConfig(variant="paper", gs_tau0=1.0, gs_tau_min=0.5, gs_lambda=1e-3)
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        assert trainer.gs_temperature() == pytest.approx(1.0)
        trainer._total_steps = 100
        assert trainer.gs_temperature() == pytest.approx(np.exp(-0.1))
        trainer._total_steps = 10_000
        assert trainer.gs_temperature() == pytest.approx(0.5)
    finally:
        env.close()


def test_relaxed_sampling_explores_at_high_tau() -> None:
    """High tau: relaxed probs ~uniform, so a strong stim logit still explores."""
    from controllers.sea_dbs.networks import sample_relaxed_action

    torch.manual_seed(0)
    logits = torch.tensor([0.0, 3.0])
    hot = []
    cold = []
    for _ in range(2000):
        relaxed, _ = gumbel_softmax_sample(logits, tau=50.0, hard=False)
        hot.append(sample_relaxed_action(relaxed))
        relaxed, _ = gumbel_softmax_sample(logits, tau=0.05, hard=False)
        cold.append(sample_relaxed_action(relaxed))
    assert 0.40 < float(np.mean(hot)) < 0.60
    # tau -> 0 approaches Gumbel-max: P(stim) = softmax(logits)[1] ~ 0.953.
    assert float(np.mean(cold)) == pytest.approx(0.953, abs=0.03)


def test_fig4_config_identical_across_variants() -> None:
    from dataclasses import asdict

    from controllers.sea_dbs.config import fig4_ravivarapu_config

    base = asdict(fig4_ravivarapu_config(variant="baseline"))
    for variant in ("baseline-pm", "baseline-gs", "paper"):
        other = asdict(fig4_ravivarapu_config(variant=variant))
        assert {k for k in base if base[k] != other[k]} == {"variant"}


def test_config_from_dict_drops_removed_knobs() -> None:
    from dataclasses import asdict

    from controllers.sea_dbs.config import config_from_dict

    raw = asdict(SEADBSConfig(variant="paper"))
    raw.update({"actor_late_episode_no_stim_boost": 2.4, "untreated_window_s": 0.1})
    assert config_from_dict(raw) == SEADBSConfig(variant="paper")
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict({**raw, "not_a_field": 1})


def test_fp16_ptq_is_a_plain_cast() -> None:
    from controllers.sea_dbs.networks import Actor
    from controllers.sea_dbs.quantization import apply_fp16_ptq

    torch.manual_seed(0)
    actor = Actor(state_dim=1, n_actions=2, hidden_size=8)
    fp16 = apply_fp16_ptq(actor)
    assert next(fp16.parameters()).dtype == torch.float16
    for p32, p16 in zip(actor.parameters(), fp16.parameters(), strict=True):
        assert torch.equal(p32.detach().half(), p16.detach())


def test_continuous_plant_integration_differs_from_disconnected() -> None:
    """Stitched timeline should move Pβ across repeated stim actions."""
    cfg = replace(
        SEADBSConfig(seed=0, biomarker_window_s=0.15),
        n_obs=1,
        max_episode_steps=3,
    )
    disc = SEA_DBSEnvAdapter(config=replace(cfg, plant_integration_mode="disconnected"))
    cont = SEA_DBSEnvAdapter(config=replace(cfg, plant_integration_mode="continuous"))
    try:
        disc.reset(seed=0)
        cont.reset(seed=0)
        disc_vals: list[float] = []
        cont_vals: list[float] = []
        for _ in range(3):
            _obs, _r, _t, _tr, info_d = disc.step(1)
            _obs, _r, _t, _tr, info_c = cont.step(1)
            disc_vals.append(float(info_d["p_beta_raw"]))
            cont_vals.append(float(info_c["p_beta_raw"]))
        assert len(set(round(v, 4) for v in disc_vals)) == 1
        assert len(set(round(v, 4) for v in cont_vals)) >= 2
    finally:
        disc.close()
        cont.close()


@pytest.mark.parametrize("variant", ["baseline", "paper"])
def test_logits_critic_stores_the_scored_action_vector(variant: str) -> None:
    """``critic_action_input='logits'`` (Mehregan et al. DDPG backbone): the
    Baseline plays argmax of its noisy logits and stores that vector; GS variants
    store their Eq. (11) relaxed sample. Updates run on those stored vectors."""

    class _FakePlant:
        def reset(self, seed=None):
            return self

        def integrate(self, duration_s, dbs_spec=None, *, record_spikes=True):
            from types import SimpleNamespace

            stim = bool(dbs_spec is not None and (dbs_spec.frequency_hz or 0.0) > 0.0)
            return SimpleNamespace(p_beta=150.0 if stim else 220.0)

        def close(self):
            return None

    cfg = SEADBSConfig(
        variant=variant, critic_action_input="logits", logit_noise_std=1.0
    ).for_smoke(episodes=2, max_steps=8)
    env = SEA_DBSEnvAdapter(plant=_FakePlant(), config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        trainer.train_episodes()
        n = len(trainer.buffer)
        vectors = trainer.buffer._logits[:n]
        actions = trainer.buffer._actions[:n]
        if variant == "baseline":
            assert np.array_equal(np.argmax(vectors, axis=1), actions)
        else:
            assert np.allclose(vectors.sum(axis=1), 1.0, atol=1e-5)
        assert trainer._update_count > 0
    finally:
        env.close()


def test_unknown_critic_action_input_rejected() -> None:
    with pytest.raises(ValueError):
        SEADBSConfig(critic_action_input="bogus").with_variant_defaults()  # type: ignore[arg-type]


def test_table2_reseed_resets_plant_every_n_steps(tmp_path) -> None:
    """Table II: the plant is reset with the next seed every n steps (Eq. (5) window re-prefilled)."""
    from types import SimpleNamespace

    from controllers.sea_dbs.checkpoint import save_checkpoint
    from controllers.sea_dbs.eval import evaluate_reseeded
    from controllers.sea_dbs.networks import Actor, Critic

    resets: list[int | None] = []

    class _FakePlant:
        def reset(self, seed: int | None = None) -> None:
            resets.append(seed)

        def integrate(self, duration_s, dbs_spec=None, *, record_spikes: bool = True):
            return SimpleNamespace(p_beta=400.0)

        def close(self) -> None:
            return None

    cfg = SEADBSConfig(seed=3, observation_scale=1000.0, n_obs=4, prefill_obs_window=True)
    actor = Actor(state_dim=cfg.state_dim, n_actions=cfg.n_actions, hidden_size=cfg.hidden_size)
    critic = Critic(state_dim=cfg.state_dim, n_actions=cfg.n_actions, hidden_size=cfg.hidden_size)
    ckpt = save_checkpoint(tmp_path / "a.pt", actor=actor, critic=critic, config=cfg)
    out = evaluate_reseeded(ckpt, reseed_every=5, horizon=20, plant=_FakePlant(), config=cfg)
    assert resets == [3, 4, 5, 6]
    assert out["block_seeds"] == [3, 4, 5, 6]
    assert len(out["p_beta_norm"]) == 20
    assert out["avg_p_beta_norm"] == pytest.approx(0.4)
    assert out["reward_of_avg_psd"] == pytest.approx(sea_dbs_reward(0.4))
    with pytest.raises(ValueError):
        evaluate_reseeded(ckpt, reseed_every=7, horizon=20, plant=_FakePlant(), config=cfg)
