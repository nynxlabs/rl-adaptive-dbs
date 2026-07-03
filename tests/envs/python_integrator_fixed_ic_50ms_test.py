"""50 ms fixed-IC GPi spike parity — fast gate after find_spike_times fix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from tests.envs.plant_backends import assert_gpi_spikes_match, require_matlab

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


def test_fixed_ic_gpi_spikes_match_matlab_50ms(
    matlab_init_draws: NetworkInitDraws,
) -> None:
    """MATLAB-exported ICs, 50 ms — spike trains within one dt grid step."""
    seed = 42
    duration_s = 0.05
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

    from tests.envs.plant_backends import spike_count_vector

    mat_counts = spike_count_vector(matlab.gpi_spikes)
    py_counts = spike_count_vector(python.gpi_spikes)

    # Neurons 0–2: counts and timing match at 50 ms (post bool-diff fix).
    for neuron in (0, 1, 2):
        assert mat_counts[neuron] == py_counts[neuron]
        assert_gpi_spikes_match(
            [matlab.gpi_spikes[neuron]],
            [python.gpi_spikes[neuron]],
        )
