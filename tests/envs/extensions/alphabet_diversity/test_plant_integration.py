"""Plant integration helpers and MehreganEnv continuous mode."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from envs.mehregan.extensions.alphabet_diversity.config import WithinStepEnvConfig
from envs.mehregan.extensions.alphabet_diversity.env import WithinStepMehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.extensions.alphabet_diversity.plant_integration import stitch_idbs
from envs.plant.config import PlantConfig
from envs.plant.python_backend import PythonPlant


def test_stitch_idbs_places_segments_after_onset() -> None:
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=45.0, step_duration_s=2.0, dt_ms=0.02, skip_regular=True
    )
    trace = stitch_idbs(
        duration_s=6.0,
        dt_ms=0.02,
        onset_sim_s=2.0,
        segment_actions=[0, 1],
        alphabet=alphabet,
        rl_step_s=2.0,
    )
    seg0 = alphabet.idbs_for_action(0)
    onset_idx = int(round(2.0 * 1000.0 / 0.02))
    step_samples = int(round(2.0 * 1000.0 / 0.02))
    assert np.any(trace[:onset_idx] == 0.0)
    assert np.array_equal(trace[onset_idx : onset_idx + seg0.size], seg0)


@pytest.mark.slow
def test_continuous_mode_differs_from_disconnected_on_repeat_action() -> None:
    """Same action twice: disconnected repeats Pβ; continuous does not."""
    seed = 0
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=45.0,
        step_duration_s=2.0,
        dt_ms=0.02,
        skip_regular=True,
    )
    plant = PythonPlant(config=PlantConfig(dt_ms=0.02))
    cfg = WithinStepEnvConfig(
        step_duration_s=2.0,
        max_episode_steps=3,
        state_mode="scalar",
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
        skip_regular=True,
        plant_dt_ms=0.02,
    )
    disc = WithinStepMehreganEnv(
        plant=plant,
        config=cfg,
        alphabet=alphabet,
    )
    cont = WithinStepMehreganEnv(
        plant=PythonPlant(config=PlantConfig(dt_ms=0.02)),
        config=replace(cfg, plant_integration_mode="continuous"),
        alphabet=alphabet,
    )
    action = 3
    disc.reset(seed=seed)
    _, _, _, _, d1 = disc.step(action)
    _, _, _, _, d2 = disc.step(action)
    cont.reset(seed=seed)
    _, _, _, _, c1 = cont.step(action)
    _, _, _, _, c2 = cont.step(action)

    assert d1["p_beta_raw"] == pytest.approx(d2["p_beta_raw"])
    assert c1["p_beta_raw"] != pytest.approx(c2["p_beta_raw"])
    assert c1["plant_integration_mode"] == "continuous"
    disc.close()
    cont.close()


def test_matlab_plant_rejects_continuous_mode() -> None:
    from envs.plant.matlab_backend import MatlabPlant

    with pytest.raises(ValueError, match="PythonPlant"):
        WithinStepMehreganEnv(
            plant=MatlabPlant(),
            config=WithinStepEnvConfig(plant_integration_mode="continuous"),
        )
