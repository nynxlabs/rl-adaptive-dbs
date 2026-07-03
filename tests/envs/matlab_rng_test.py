"""MATLAB-compatible uniform RNG helpers."""

from __future__ import annotations

import numpy as np

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
