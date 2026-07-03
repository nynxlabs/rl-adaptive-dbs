"""Cross-backend equivalence: MATLAB reference vs native PythonPlant."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from envs.plant import DbsSpec, IntegrateResult, MatlabPlant, PlantConfig, PythonPlant
from tests.envs.plant_backends import (
    assert_gpi_spikes_match,
    assert_p_beta_match,
    require_matlab,
    spike_count_vector,
)

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
def python_plant() -> Iterator[PythonPlant]:
    with PythonPlant(PlantConfig()) as plant:
        yield plant


def _integrate(
    plant: MatlabPlant | PythonPlant,
    *,
    seed: int,
    duration_s: float,
    dbs_spec: DbsSpec | None = None,
) -> IntegrateResult:
    return plant.reset(seed=seed).integrate(duration_s, dbs_spec or DbsSpec.none())


@pytest.mark.parametrize("seed", [42, 7, 3])
def test_gpi_spikes_match_matlab_no_dbs(
    matlab_plant: MatlabPlant,
    python_plant: PythonPlant,
    seed: int,
) -> None:
    matlab = _integrate(matlab_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    python = _integrate(python_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    assert_gpi_spikes_match(matlab.gpi_spikes, python.gpi_spikes)


@pytest.mark.parametrize("seed", [42, 7])
def test_p_beta_matches_matlab_no_dbs(
    matlab_plant: MatlabPlant,
    python_plant: PythonPlant,
    seed: int,
) -> None:
    matlab = _integrate(matlab_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    python = _integrate(python_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    assert matlab.p_beta is not None and python.p_beta is not None
    assert_p_beta_match(matlab.p_beta, python.p_beta)


@pytest.mark.parametrize(
    ("label", "dbs_spec"),
    [
        ("cdbs-130hz", DbsSpec.from_frequency_hz(130)),
        ("periodic-45hz", DbsSpec.from_frequency_hz(45)),
    ],
)
def test_p_beta_ordering_under_dbs(
    matlab_plant: MatlabPlant,
    python_plant: PythonPlant,
    label: str,
    dbs_spec: DbsSpec,
) -> None:
    seed = 11
    matlab_none = _integrate(matlab_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    matlab_dbs = _integrate(matlab_plant, seed=seed, duration_s=2.0, dbs_spec=dbs_spec)
    python_none = _integrate(python_plant, seed=seed, duration_s=2.0, dbs_spec=DbsSpec.none())
    python_dbs = _integrate(python_plant, seed=seed, duration_s=2.0, dbs_spec=dbs_spec)

    assert matlab_none.p_beta is not None and matlab_dbs.p_beta is not None
    assert python_none.p_beta is not None and python_dbs.p_beta is not None

    assert_p_beta_match(matlab_none.p_beta, python_none.p_beta)
    assert_p_beta_match(matlab_dbs.p_beta, python_dbs.p_beta)

    # CD-DBS / periodic stimulation should not increase beta vs none (paper ordering).
    assert matlab_dbs.p_beta < matlab_none.p_beta, label
    assert python_dbs.p_beta < python_none.p_beta, label


def test_spike_counts_match_at_seed_42(
    matlab_plant: MatlabPlant,
    python_plant: PythonPlant,
) -> None:
    """Regression guard — per-neuron GPi spike counts at seed=42 (2 s, no DBS)."""
    matlab = _integrate(matlab_plant, seed=42, duration_s=2.0, dbs_spec=DbsSpec.none())
    python = _integrate(python_plant, seed=42, duration_s=2.0, dbs_spec=DbsSpec.none())
    assert spike_count_vector(matlab.gpi_spikes) == spike_count_vector(python.gpi_spikes)
    assert matlab.p_beta is not None and python.p_beta is not None
    assert_p_beta_match(matlab.p_beta, python.p_beta)
