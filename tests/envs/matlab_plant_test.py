"""Plant bridge tests — parametrized MATLAB and Python backends."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, PythonPlant
from envs.plant.spikes import spike_counts
from tests.envs.plant_backends import assert_gpi_spikes_match


def _spikes_allclose(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    assert_gpi_spikes_match(a, b)


@pytest.mark.slow
def test_integrate_returns_gpi_spikes(
    module_plant: MatlabPlant | PythonPlant,
) -> None:
    result = module_plant.reset(seed=42).integrate(2.0, DbsSpec.none())
    assert len(result.gpi_spikes) == 10
    assert all(times.ndim == 1 for times in result.gpi_spikes)
    assert result.dt_ms == pytest.approx(0.01)
    assert result.pd == 1
    assert result.info["dbs_freq_hz"] == pytest.approx(0.0)
    assert sum(spike_counts(result.gpi_spikes)) > 0


@pytest.mark.slow
def test_integrate_reproducible_with_seed(
    module_plant: MatlabPlant | PythonPlant,
) -> None:
    first = module_plant.reset(seed=7).integrate(2.0, DbsSpec.none())
    second = module_plant.reset(seed=7).integrate(2.0, DbsSpec.none())
    _spikes_allclose(first.gpi_spikes, second.gpi_spikes)


@pytest.mark.slow
def test_integrate_differs_without_seed(
    module_plant: MatlabPlant | PythonPlant,
) -> None:
    first = module_plant.reset(seed=11).integrate(2.0, DbsSpec.none())
    second = module_plant.reset(seed=12).integrate(2.0, DbsSpec.none())
    assert spike_counts(first.gpi_spikes).tolist() != spike_counts(
        second.gpi_spikes
    ).tolist()


@pytest.mark.slow
def test_no_dbs_matches_pick_dbs_freq_one(
    module_plant: MatlabPlant | PythonPlant,
) -> None:
    from_none = module_plant.reset(seed=3).integrate(2.0, DbsSpec.none())
    from_index = module_plant.reset(seed=3).integrate(
        2.0, DbsSpec(pick_dbs_freq=1)
    )
    _spikes_allclose(from_none.gpi_spikes, from_index.gpi_spikes)


def test_dbs_spec_frequency_mapping() -> None:
    assert DbsSpec.from_frequency_hz(0).pick_dbs_freq == 1
    assert DbsSpec.from_frequency_hz(130).pick_dbs_freq == 27
    assert DbsSpec.from_frequency_hz(45).pick_dbs_freq == 10
