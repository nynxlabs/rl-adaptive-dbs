"""First GPi spike time with MATLAB-exported ICs — fast parity smoke (integrator drift is later)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from tests.envs.plant_backends import require_matlab

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plant_init_seed42.npz"

pytestmark = [pytest.mark.matlab, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _require_matlab() -> None:
    require_matlab()


@pytest.fixture(scope="module")
def matlab_init_draws() -> NetworkInitDraws:
    if not FIXTURE.is_file():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return NetworkInitDraws.from_npz(FIXTURE)


def test_first_gpi_spike_time_matches_matlab_with_fixed_ics(
    matlab_init_draws: NetworkInitDraws,
) -> None:
    """First crossing of -20 mV should match when ICs/wiring are identical (seed=42)."""
    seed = 42
    duration_s = 0.05  # first GPi spikes occur within ~20 ms; avoids full 2 s Python cost
    cfg = PlantConfig()

    with MatlabPlant(cfg) as matlab_plant:
        matlab = matlab_plant.reset(seed=seed).integrate(duration_s, DbsSpec.none())

    rng = np.random.default_rng(seed)
    python = integrate_network(
        config=cfg,
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=rng,
        iteration=1,
        seed=seed,
        init_draws=matlab_init_draws,
    )

    for neuron in range(cfg.neurons_per_region):
        mat_times = matlab.gpi_spikes[neuron]
        py_times = python.gpi_spikes[neuron]
        if mat_times.size == 0:
            assert py_times.size == 0, f"GPi neuron {neuron}: Python spiked, MATLAB did not"
            continue
        assert py_times.size > 0, f"GPi neuron {neuron}: MATLAB spiked, Python did not"
        np.testing.assert_allclose(
            py_times[0],
            mat_times[0],
            rtol=0.0,
            atol=1e-9,
            err_msg=f"GPi neuron {neuron} first spike time differs",
        )
