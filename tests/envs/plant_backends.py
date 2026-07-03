"""Shared helpers for MATLAB / Python plant backend tests (not collected)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pytest

from envs.plant import DbsSpec, IntegrateResult, MatlabPlant, PlantConfig, PythonPlant, p_beta
from envs.plant.spikes import spike_counts

PlantBackendName = Literal["matlab", "python"]

BACKEND_NAMES: tuple[PlantBackendName, ...] = ("matlab", "python")

# Equivalence gates (docs/plant.md §8, native-plant-port.md §5).
# Spike times are on the shared 0.01 ms grid — target exact match (0 ms atol).
SPIKE_TIME_ATOL_MS: float = 0.0
# One Kumaravelu dt step (0.01 ms); short-segment tests may allow a few steps drift.
SPIKE_TIME_ATOL_S: float = 1e-5
# Normalized Mehregan-scale P_beta (same multitaper path for both backends).
P_BETA_REL_TOL: float = 0.01
P_BETA_ABS_TOL: float = 0.01

# Phase B drift (2026-07-03 bisection, seed=42, fixed MATLAB ICs):
# - GPi trains match through 69.07 ms (4 spikes, neuron 0).
# - Python 5th spike at 69.07 ms by 69.08 ms; MATLAB 5th at 69.12 ms — dynamics drift
#   at integrator step ~6907, not RNG (uniform MT19937 matches; randn needs fixtures).
# - 2 s: mat counts [102,…] vs py [100,…]; p_beta rel err ≫ 1%.
# PythonPlant.reset loads tests/fixtures/plant_init_seed{seed}.npz when present.
PHASE_B_EQUIVALENCE_XFAIL_REASON = (
    "Integrator dynamics drift from ~69 ms (fixed ICs); 2 s / cross-backend open"
)


def matlab_available() -> bool:
    from tests.conftest import matlab_available as _check

    return _check()


def require_matlab() -> None:
    if not matlab_available():
        pytest.skip("MATLAB not installed or not licensed")


@contextmanager
def plant_session(backend: PlantBackendName) -> Iterator[MatlabPlant | PythonPlant]:
    if backend == "matlab":
        require_matlab()
        with MatlabPlant(PlantConfig()) as plant:
            yield plant
    elif backend == "python":
        with PythonPlant(PlantConfig()) as plant:
            yield plant
    else:
        msg = f"unknown backend: {backend!r}"
        raise ValueError(msg)


def integrate_segment(
    backend: PlantBackendName,
    *,
    seed: int,
    duration_s: float,
    dbs_spec: DbsSpec | None = None,
) -> IntegrateResult:
    with plant_session(backend) as plant:
        return plant.reset(seed=seed).integrate(duration_s, dbs_spec or DbsSpec.none())


def assert_gpi_spikes_match(
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    *,
    atol_ms: float = SPIKE_TIME_ATOL_MS,
) -> None:
    """Assert per-neuron GPi spike trains match (same integration grid)."""
    atol_s = max(atol_ms / 1000.0, SPIKE_TIME_ATOL_S) if atol_ms == 0.0 else atol_ms / 1000.0
    assert len(reference) == len(candidate)
    for neuron_index, (ref_times, cand_times) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        np.testing.assert_allclose(
            ref_times,
            cand_times,
            rtol=0.0,
            atol=atol_s,
            err_msg=f"GPi neuron {neuron_index} spike times differ",
        )


def assert_p_beta_match(
    reference: float,
    candidate: float,
    *,
    rel_tol: float = P_BETA_REL_TOL,
    abs_tol: float = P_BETA_ABS_TOL,
) -> None:
    assert reference == pytest.approx(candidate, rel=rel_tol, abs=abs_tol)


def p_beta_from_spikes(result: IntegrateResult) -> float:
    return p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        segment_duration_s=result.duration_s,
    )


def spike_count_vector(gpi_spikes: list[np.ndarray]) -> list[int]:
    return spike_counts(gpi_spikes).tolist()
