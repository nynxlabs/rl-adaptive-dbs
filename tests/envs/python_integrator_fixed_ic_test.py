"""Integrator parity with MATLAB-exported initialization draws (isolates RNG from dynamics)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from tests.envs.plant_backends import (
    assert_gpi_spikes_match,
    assert_p_beta_match,
    require_matlab,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plant_init_seed42.npz"

pytestmark = [
    pytest.mark.matlab,
    pytest.mark.slow,
]


@pytest.fixture(scope="module", autouse=True)
def _require_matlab() -> None:
    require_matlab()


@pytest.fixture(scope="module")
def matlab_plant() -> Iterator[MatlabPlant]:
    with MatlabPlant(PlantConfig()) as plant:
        yield plant


@pytest.fixture(scope="module")
def matlab_init_draws() -> NetworkInitDraws:
    if not FIXTURE.is_file():
        pytest.skip(f"fixture missing: {FIXTURE} (run scripts/export_plant_init_draws.py)")
    return NetworkInitDraws.from_npz(FIXTURE)


def test_integrator_matches_matlab_with_fixed_init_draws(
    matlab_plant: MatlabPlant,
    matlab_init_draws: NetworkInitDraws,
) -> None:
    """Same ICs/wiring as MATLAB seed=42 — full GPi train parity gate."""
    seed = 42
    duration_s = 2.0
    dbs_spec = DbsSpec.none()

    matlab = matlab_plant.reset(seed=seed).integrate(duration_s, dbs_spec)

    rng = np.random.default_rng(seed)
    python = integrate_network(
        config=PlantConfig(),
        duration_s=duration_s,
        dbs_spec=dbs_spec,
        record_spikes=True,
        rng=rng,
        iteration=1,
        seed=seed,
        init_draws=matlab_init_draws,
    )

    assert matlab.p_beta is not None and python.p_beta is not None
    assert_gpi_spikes_match(matlab.gpi_spikes, python.gpi_spikes)
    assert_p_beta_match(matlab.p_beta, python.p_beta)


def test_python_plant_accepts_fixed_init_draws(
    matlab_init_draws: NetworkInitDraws,
) -> None:
    rng = np.random.default_rng(42)
    result = integrate_network(
        config=PlantConfig(),
        duration_s=0.01,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=rng,
        iteration=1,
        seed=42,
        init_draws=matlab_init_draws,
    )
    assert len(result.gpi_spikes) == PlantConfig().neurons_per_region
