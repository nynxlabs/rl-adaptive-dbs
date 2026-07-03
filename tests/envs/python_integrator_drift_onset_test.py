"""Integrator parity checkpoints — fixed ICs, GPi neuron 0 (seed 42)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from tests.envs.plant_backends import assert_gpi_spikes_match, require_matlab

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plant_init_seed42.npz"

MATCH_CHECKPOINTS_MS: tuple[float, ...] = (69.07, 69.08)

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
) -> list:
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


@pytest.mark.parametrize("duration_ms", MATCH_CHECKPOINTS_MS)
def test_gpi_trains_match_matlab_checkpoint(
    matlab_init_draws: NetworkInitDraws,
    duration_ms: float,
) -> None:
    """GPi neuron 0 — regression gates around the old ~69 ms drift window."""
    duration_s = duration_ms / 1000.0
    neuron = 0
    with MatlabPlant(PlantConfig()) as plant:
        matlab = plant.reset(seed=42).integrate(duration_s, DbsSpec.none())
    python_spikes = _integrate_python(matlab_init_draws, duration_s)
    assert_gpi_spikes_match(
        [matlab.gpi_spikes[neuron]],
        [python_spikes[neuron]],
    )
