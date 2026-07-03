"""Integrator drift onset — fixed ICs, GPi neuron 0 (2026-07-03 bisection)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from tests.envs.plant_backends import assert_gpi_spikes_match, require_matlab

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plant_init_seed42.npz"

# Bisected 2026-07-03: trains match through 69.07 ms; Python 5th spike appears by 69.08 ms
# while MATLAB still has four spikes until ~69.12 ms (integrator dynamics, not RNG).
DRIFT_ONSET_MS: float = 69.08
MATCH_UNTIL_MS: float = 69.07

pytestmark = [pytest.mark.matlab, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _require_matlab() -> None:
    require_matlab()


@pytest.fixture(scope="module")
def matlab_init_draws() -> NetworkInitDraws:
    if not FIXTURE.is_file():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return NetworkInitDraws.from_npz(FIXTURE)


def _integrate_python(
    draws: NetworkInitDraws,
    duration_s: float,
) -> list[np.ndarray]:
    result = integrate_network(
        config=PlantConfig(),
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=np.random.default_rng(42),
        iteration=1,
        seed=42,
        init_draws=draws,
    )
    return result.gpi_spikes


def test_gpi_trains_match_through_69_07ms(matlab_init_draws: NetworkInitDraws) -> None:
    """GPi neuron 0 — last matched segment before fifth-spike divergence."""
    duration_s = MATCH_UNTIL_MS / 1000.0
    neuron = 0
    with MatlabPlant(PlantConfig()) as plant:
        matlab = plant.reset(seed=42).integrate(duration_s, DbsSpec.none())
    python_spikes = _integrate_python(matlab_init_draws, duration_s)
    assert_gpi_spikes_match(
        [matlab.gpi_spikes[neuron]],
        [python_spikes[neuron]],
    )


def test_gpi_train_diverges_by_69_08ms(matlab_init_draws: NetworkInitDraws) -> None:
    duration_s = DRIFT_ONSET_MS / 1000.0
    with MatlabPlant(PlantConfig()) as plant:
        matlab = plant.reset(seed=42).integrate(duration_s, DbsSpec.none())
    python_spikes = _integrate_python(matlab_init_draws, duration_s)
    assert len(matlab.gpi_spikes[0]) == 4
    assert len(python_spikes[0]) == 5
    assert python_spikes[0][4] == pytest.approx(0.06907, abs=1e-5)
