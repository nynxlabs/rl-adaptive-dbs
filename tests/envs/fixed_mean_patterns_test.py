"""Tests for the fixed-mean-frequency pulse pattern action space (TASK-84).

Covers:
- Mean-rate preservation across all patterns.
- Pattern-0 equivalence to the regular train (create_dbs_current).
- FixedMeanPatternAlphabet → DbsSpec round-trip.
- make_alphabet factory.
- baseline_action in pattern mode.
- init_baseline_for_variant in pattern mode.
- Env smoke with MockPlant in pattern mode.
- PythonPlant integration (DbsSpec.idbs integrator dispatch).
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.mehregan.baselines import baseline_action
from envs.mehregan.config import MehreganEnvConfig, make_alphabet
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.patterns import PatternAlphabet
from envs.plant import PythonPlant
from envs.plant.dbs import DbsSpec, create_dbs_current
from controllers.ddpg.config import DDPGConfig, init_baseline_for_variant
from tests.envs.mock_plant import MockPlant


# ---------------------------------------------------------------------------
# Alphabet construction
# ---------------------------------------------------------------------------


class TestFixedMeanPatternAlphabet:
    """Unit tests for FixedMeanPatternAlphabet."""

    def test_n_actions_41(self) -> None:
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        assert alpha.n_actions == 41

    def test_all_patterns_same_pulse_count(self) -> None:
        """Mean-rate preservation: every pattern has the same number of pulses."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        counts = [alpha.pulse_count(i) for i in range(alpha.n_actions)]
        unique = set(counts)
        assert len(unique) == 1, f"expected 1 unique pulse count, got {unique}"
        # 45 Hz over 2 s on 0.01 ms grid (inclusive endpoints) → 91 pulses
        assert counts[0] == 91

    def test_pattern0_matches_regular_train(self) -> None:
        """Pattern 0 should be byte-identical to create_dbs_current(mean_hz)."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        pattern_trace = alpha.idbs_for_pattern(0)
        regular = create_dbs_current(
            45.0,
            tmax_ms=alpha.step_duration_s * 1000.0,
            dt_ms=alpha.dt_ms,
        )
        np.testing.assert_array_equal(pattern_trace, regular)

    def test_irregular_patterns_differ_from_regular(self) -> None:
        """At least some irregular patterns should differ from the regular train."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        regular = alpha.idbs_for_pattern(0)
        n_different = sum(
            1
            for i in range(1, alpha.n_actions)
            if not np.array_equal(alpha.idbs_for_pattern(i), regular)
        )
        # With 40 jittered patterns, most should differ (some may coincidentally
        # match if jitter is zero, but with default 1/3 jitter that's unlikely).
        assert n_different >= 30, f"only {n_different}/40 irregular patterns differ"

    def test_pattern_reproducibility(self) -> None:
        """Same (mean_hz, index) always produces the same trace."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        for i in range(alpha.n_actions):
            a = alpha.idbs_for_pattern(i)
            b = alpha.idbs_for_pattern(i)
            np.testing.assert_array_equal(a, b)

    def test_different_mean_hz(self) -> None:
        """30 Hz alphabet should have the same pulse count on every pattern."""
        alpha = FixedMeanPatternAlphabet(mean_hz=30.0)
        counts = [alpha.pulse_count(i) for i in range(alpha.n_actions)]
        expected = alpha.pulse_count(0)
        assert all(c == expected for c in counts)

    def test_to_dbs_spec_has_idbs(self) -> None:
        """to_dbs_spec should attach the precomputed trace."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        spec = alpha.to_dbs_spec(5)
        assert spec.idbs is not None
        assert spec.mean_hz == 45.0
        np.testing.assert_array_equal(spec.idbs, alpha.idbs_for_pattern(5))

    def test_to_dbs_spec_pattern0(self) -> None:
        """Pattern 0 → DbsSpec should have mean_hz and idbs."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        spec = alpha.to_dbs_spec(0)
        assert spec.idbs is not None
        assert spec.mean_hz == 45.0
        assert spec.frequency_hz == 45.0

    def test_action_for_frequency_hz_mean(self) -> None:
        """action_for_frequency_hz(mean_hz) → 0."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        assert alpha.action_for_frequency_hz(45.0) == 0

    def test_action_for_frequency_hz_non_mean_raises(self) -> None:
        """action_for_frequency_hz(non-mean) raises ValueError."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        with pytest.raises(ValueError, match="no fixed-mean pattern"):
            alpha.action_for_frequency_hz(130.0)

    def test_action_for_dbs_spec_mean(self) -> None:
        """action_for_dbs_spec with mean Hz → 0."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        spec = DbsSpec.from_frequency_hz(45.0)
        assert alpha.action_for_dbs_spec(spec) == 0

    def test_action_for_dbs_spec_non_mean_raises(self) -> None:
        """action_for_dbs_spec with non-mean frequency raises ValueError."""
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        spec = DbsSpec.from_frequency_hz(130.0)
        with pytest.raises(ValueError, match="no fixed-mean pattern"):
            alpha.action_for_dbs_spec(spec)


# ---------------------------------------------------------------------------
# make_alphabet factory
# ---------------------------------------------------------------------------


class TestMakeAlphabet:
    """Tests for config.make_alphabet factory."""

    def test_scalar_frequency_default(self) -> None:
        config = MehreganEnvConfig()
        alpha = make_alphabet(config)
        assert isinstance(alpha, PatternAlphabet)

    def test_fixed_mean_pattern(self) -> None:
        config = MehreganEnvConfig(action_space_mode="fixed_mean_pattern")
        alpha = make_alphabet(config)
        assert isinstance(alpha, FixedMeanPatternAlphabet)
        assert alpha.mean_hz == 45.0

    def test_fixed_mean_pattern_custom_hz(self) -> None:
        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=30.0,
        )
        alpha = make_alphabet(config)
        assert isinstance(alpha, FixedMeanPatternAlphabet)
        assert alpha.mean_hz == 30.0

    def test_fixed_mean_pattern_respects_step_duration(self) -> None:
        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            step_duration_s=1.6,
        )
        alpha = make_alphabet(config)
        assert isinstance(alpha, FixedMeanPatternAlphabet)
        assert alpha.step_duration_s == 1.6

    def test_unknown_action_space_mode_raises(self) -> None:
        config = MehreganEnvConfig(action_space_mode="bogus")
        with pytest.raises(ValueError, match="unknown action_space_mode"):
            make_alphabet(config)


# ---------------------------------------------------------------------------
# baseline_action in pattern mode
# ---------------------------------------------------------------------------


class TestBaselineActionPatternMode:
    """baseline_action with FixedMeanPatternAlphabet."""

    def test_periodic_45hz_maps_to_pattern0(self) -> None:
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        action = baseline_action("periodic-45hz", alpha)
        assert action == 0

    def test_periodic_30hz_on_30hz_alphabet(self) -> None:
        alpha = FixedMeanPatternAlphabet(mean_hz=30.0)
        action = baseline_action("periodic-30hz", alpha)
        assert action == 0

    def test_non_mean_baseline_raises(self) -> None:
        alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
        with pytest.raises(ValueError, match="no fixed-mean pattern"):
            baseline_action("cdbs-130hz", alpha)


# ---------------------------------------------------------------------------
# init_baseline_for_variant in pattern mode
# ---------------------------------------------------------------------------


class TestInitBaselinePatternMode:
    """init_baseline_for_variant with pattern mode."""

    def test_pattern_mode_returns_mean_hz_baseline(self) -> None:
        name = init_baseline_for_variant(
            "paper",
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
        )
        assert name == "periodic-45hz"

    def test_pattern_mode_30hz(self) -> None:
        name = init_baseline_for_variant(
            "init-30hz",
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=30.0,
        )
        assert name == "periodic-30hz"

    def test_scalar_mode_unchanged(self) -> None:
        assert init_baseline_for_variant("paper") == "periodic-45hz"
        assert init_baseline_for_variant("init-30hz") == "periodic-30hz"


# ---------------------------------------------------------------------------
# DDPGConfig pattern mode fields
# ---------------------------------------------------------------------------


class TestDDPGConfigPatternMode:
    """DDPGConfig action_space_mode / pattern_mean_hz fields."""

    def test_default_scalar(self) -> None:
        cfg = DDPGConfig()
        assert cfg.action_space_mode == "scalar_frequency"
        assert cfg.pattern_mean_hz == 45.0
        assert cfg.init_baseline == "periodic-45hz"

    def test_pattern_mode(self) -> None:
        cfg = DDPGConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
        )
        assert cfg.init_baseline == "periodic-45hz"

    def test_pattern_mode_30hz(self) -> None:
        cfg = DDPGConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=30.0,
        )
        assert cfg.init_baseline == "periodic-30hz"

    def test_init_30hz_variant_syncs_pattern_mean_hz(self) -> None:
        """TASK-85 B2: variant slug implies 30 Hz even when pattern_mean_hz defaults to 45."""
        cfg = DDPGConfig(
            variant="init-30hz",
            action_space_mode="fixed_mean_pattern",
        )
        assert cfg.effective_pattern_mean_hz == 30.0
        assert cfg.init_baseline == "periodic-30hz"
        synced = cfg.with_variant_defaults()
        assert synced.pattern_mean_hz == 30.0


# ---------------------------------------------------------------------------
# Env smoke test with pattern mode
# ---------------------------------------------------------------------------


class TestMehreganEnvPatternMode:
    """Smoke tests: MehreganEnv works with FixedMeanPatternAlphabet via config."""

    @pytest.fixture
    def pattern_env(self) -> MehreganEnv:
        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
            max_episode_steps=3,
        )
        env = MehreganEnv(plant=MockPlant(), config=config)
        yield env
        env.close()

    def test_env_has_pattern_alphabet(self, pattern_env: MehreganEnv) -> None:
        assert isinstance(pattern_env.alphabet, FixedMeanPatternAlphabet)
        assert pattern_env.action_space.n == 41

    def test_reset_works(self, pattern_env: MehreganEnv) -> None:
        obs, info = pattern_env.reset(seed=0)
        assert obs.shape == (1,)
        assert obs.dtype == np.float32
        assert "p_beta_raw" in info

    def test_step_with_pattern0(self, pattern_env: MehreganEnv) -> None:
        pattern_env.reset(seed=0)
        obs, reward, terminated, truncated, info = pattern_env.step(0)
        assert obs.shape == (1,)
        assert isinstance(reward, float)
        assert not terminated
        assert info["action"] == 0

    def test_step_with_irregular_pattern(self, pattern_env: MehreganEnv) -> None:
        pattern_env.reset(seed=0)
        obs, reward, terminated, truncated, info = pattern_env.step(5)
        assert obs.shape == (1,)
        assert info["action"] == 5

    def test_full_episode(self, pattern_env: MehreganEnv) -> None:
        """Run a full 3-step episode without errors."""
        pattern_env.reset(seed=0)
        for step in range(3):
            obs, reward, terminated, truncated, info = pattern_env.step(step % 5)
            if terminated or truncated:
                break
        # Should complete without error

    def test_baseline_rollout_in_pattern_mode(self) -> None:
        """run_baseline_rollout with pattern-mode env and periodic-45hz baseline."""
        from envs.mehregan.baselines import run_baseline_rollout

        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
            max_episode_steps=2,
        )
        env = MehreganEnv(plant=MockPlant(), config=config)
        try:
            result = run_baseline_rollout(env, "periodic-45hz", seed=0)
            assert result["baseline"] == "periodic-45hz"
            assert result["steps"] == 2
            assert result["action"] == 0  # pattern 0 = regular at 45 Hz
        finally:
            env.close()


# ---------------------------------------------------------------------------
# PythonPlant integration (idbs dispatch path)
# ---------------------------------------------------------------------------


class TestMehreganEnvPatternModePythonPlant:
    """Pattern-mode env steps through PythonPlant with precomputed idbs traces."""

    @pytest.mark.slow
    def test_step_pattern0_and_irregular_on_python_plant(self) -> None:
        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
            max_episode_steps=2,
        )
        with PythonPlant() as plant:
            env = MehreganEnv(plant=plant, config=config)
            try:
                assert isinstance(env.alphabet, FixedMeanPatternAlphabet)
                obs, info = env.reset(seed=42)
                assert obs.shape == (1,)
                assert info["p_beta_raw"] > 0.0

                _, reward0, terminated, truncated, step0 = env.step(0)
                assert isinstance(reward0, float)
                assert step0["p_beta_raw"] > 0.0
                assert not terminated
                assert not truncated

                _, reward5, terminated2, truncated2, step5 = env.step(5)
                assert isinstance(reward5, float)
                assert step5["p_beta_raw"] > 0.0
                assert step5["action"] == 5
                assert terminated2 or truncated2
            finally:
                env.close()


# ---------------------------------------------------------------------------
# train() default env (TASK-85 B1)
# ---------------------------------------------------------------------------


class TestTrainDefaultEnv:
    """``default_train_env`` must use PythonPlant in pattern mode."""

    def test_pattern_mode_uses_python_plant(self) -> None:
        from controllers.ddpg import default_train_env
        from controllers.ddpg.config import DDPGConfig
        from envs.plant.python_backend import PythonPlant

        env = default_train_env(
            DDPGConfig(action_space_mode="fixed_mean_pattern", pattern_mean_hz=45.0),
        )
        try:
            assert isinstance(env._plant, PythonPlant)
            assert isinstance(env.alphabet, FixedMeanPatternAlphabet)
        finally:
            env.close()

    def test_scalar_mode_default_plant(self) -> None:
        from controllers.ddpg import default_train_env
        from controllers.ddpg.config import DDPGConfig
        from envs.plant.matlab_backend import MatlabPlant

        env = default_train_env(DDPGConfig())
        try:
            assert isinstance(env._plant, MatlabPlant)
        finally:
            env.close()

    def test_init_30hz_pattern_mode_env_and_baseline(self) -> None:
        """TASK-85 B2: default_train_env + init_baseline stay at 30 Hz for init-30hz."""
        from controllers.ddpg import default_train_env
        from controllers.ddpg.config import DDPGConfig

        cfg = DDPGConfig(
            variant="init-30hz",
            action_space_mode="fixed_mean_pattern",
        )
        assert cfg.init_baseline == "periodic-30hz"
        env = default_train_env(cfg)
        try:
            assert env.config.pattern_mean_hz == 30.0
            assert isinstance(env.alphabet, FixedMeanPatternAlphabet)
            assert env.alphabet.mean_hz == 30.0
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Env with explicit alphabet still works
# ---------------------------------------------------------------------------


class TestEnvExplicitAlphabet:
    """Passing an explicit alphabet still overrides config."""

    def test_explicit_alphabet_overrides_config(self) -> None:
        config = MehreganEnvConfig(
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=30.0,
        )
        # Pass explicit scalar alphabet → should override config's pattern mode
        env = MehreganEnv(
            plant=MockPlant(),
            config=config,
            alphabet=PatternAlphabet(),
        )
        try:
            assert isinstance(env.alphabet, PatternAlphabet)
            assert env.action_space.n == 41
        finally:
            env.close()
