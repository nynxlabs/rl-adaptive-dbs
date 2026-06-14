"""MATLAB plant bridge (Kumaravelu et al., 2016)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PlantConfig
from envs.plant.spikes import spike_counts


def _spikes_allclose(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        np.testing.assert_allclose(left, right, rtol=0, atol=0)


@pytest.fixture(scope="module")
def matlab_plant() -> MatlabPlant:
    with MatlabPlant(PlantConfig()) as plant:
        yield plant


@pytest.mark.matlab
def test_integrate_returns_gpi_spikes(matlab_plant: MatlabPlant) -> None:
    result = matlab_plant.reset(seed=42).integrate(2.0, DbsSpec.none())
    assert len(result.gpi_spikes) == 10
    assert all(times.ndim == 1 for times in result.gpi_spikes)
    assert result.dt_ms == pytest.approx(0.01)
    assert result.pd == 1
    assert result.info["dbs_freq_hz"] == pytest.approx(0.0)
    assert sum(spike_counts(result.gpi_spikes)) > 0


@pytest.mark.matlab
def test_integrate_reproducible_with_seed(matlab_plant: MatlabPlant) -> None:
    first = matlab_plant.reset(seed=7).integrate(2.0, DbsSpec.none())
    second = matlab_plant.reset(seed=7).integrate(2.0, DbsSpec.none())
    _spikes_allclose(first.gpi_spikes, second.gpi_spikes)


@pytest.mark.matlab
def test_integrate_differs_without_seed(matlab_plant: MatlabPlant) -> None:
    first = matlab_plant.reset(seed=11).integrate(2.0, DbsSpec.none())
    second = matlab_plant.reset(seed=12).integrate(2.0, DbsSpec.none())
    assert spike_counts(first.gpi_spikes).tolist() != spike_counts(
        second.gpi_spikes
    ).tolist()


@pytest.mark.matlab
def test_no_dbs_matches_pick_dbs_freq_one(matlab_plant: MatlabPlant) -> None:
    from_none = matlab_plant.reset(seed=3).integrate(2.0, DbsSpec.none())
    from_index = matlab_plant.reset(seed=3).integrate(
        2.0, DbsSpec(pick_dbs_freq=1)
    )
    _spikes_allclose(from_none.gpi_spikes, from_index.gpi_spikes)


@pytest.mark.matlab
def test_dbs_spec_frequency_mapping() -> None:
    assert DbsSpec.from_frequency_hz(0).pick_dbs_freq == 1
    assert DbsSpec.from_frequency_hz(130).pick_dbs_freq == 27
    assert DbsSpec.from_frequency_hz(45).pick_dbs_freq == 10
