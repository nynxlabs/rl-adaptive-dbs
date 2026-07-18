"""MATLAB-compatible uniform RNG helpers."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant.network.matlab_rng import MatlabRandomState, load_cached_init_draws


def test_matlab_uniform_rand_matches_random_state() -> None:
    rs = np.random.RandomState(42)
    matlab = MatlabRandomState(42)
    py = matlab.rand(50)
    ref = rs.rand(50)
    np.testing.assert_array_equal(py, ref)


def test_load_cached_init_draws_seed42() -> None:
    draws = load_cached_init_draws(42)
    assert draws is not None
    assert draws.v1.shape == (10,)


@pytest.mark.matlab
def test_cached_init_draws_gsngen_matches_matlab_export() -> None:
    """Fixture gsngen must match live CTX_BG_TH_network (plant_init_export)."""
    pytest.importorskip("matlab.engine")
    from tests.conftest import matlab_engine_available

    if not matlab_engine_available():
        pytest.skip("MATLAB Python engine not installed (uv sync --group matlab)")

    from scripts.export_plant_init_draws import export_matlab_init_draws

    draws = load_cached_init_draws(42)
    assert draws is not None
    fresh = export_matlab_init_draws(seed=42)
    np.testing.assert_allclose(draws.gsngen, fresh["gsngen"])
    np.testing.assert_allclose(draws.gsngea, fresh["gsngea"])
    np.testing.assert_allclose(draws.gsngi, fresh["gsngi"])
