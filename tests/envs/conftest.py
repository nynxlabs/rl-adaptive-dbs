"""Fixtures for env / plant integration tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from envs.plant import MatlabPlant, PlantConfig, PythonPlant
from tests.envs.plant_backends import BACKEND_NAMES, PlantBackendName, matlab_engine_available


@pytest.fixture(scope="module", params=BACKEND_NAMES, ids=list(BACKEND_NAMES))
def plant_backend(request: pytest.FixtureRequest) -> PlantBackendName:
    backend: PlantBackendName = request.param
    if backend == "matlab" and not matlab_engine_available():
        pytest.skip("MATLAB Python engine not installed (uv sync --group matlab)")
    return backend


@pytest.fixture(scope="module")
def module_plant(
    plant_backend: PlantBackendName,
) -> Iterator[MatlabPlant | PythonPlant]:
    if plant_backend == "matlab":
        with MatlabPlant(PlantConfig()) as plant:
            yield plant
    else:
        with PythonPlant(PlantConfig()) as plant:
            yield plant
