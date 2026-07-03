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
# Normalized Mehregan-scale P_beta (same multitaper path for both backends).
P_BETA_REL_TOL: float = 0.01
P_BETA_ABS_TOL: float = 0.01

# Known Phase B drift (2026-07-03): Python uses numpy.random.Generator (PCG64) while
# MATLAB rng(seed) uses the legacy twister; integrator parity is still open. Reviewer
# should log measured drift here until gates pass — do not relax SPIKE_TIME_ATOL_MS
# without a documented detection-threshold exception in native-plant-port.md.
PHASE_B_EQUIVALENCE_XFAIL_REASON = (
    "PythonPlant integrator + MATLAB RNG parity pending "
    "(fixed-IC test still drifts GPi spikes; see python_integrator_fixed_ic_test.py)"
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
    assert len(reference) == len(candidate)
    for neuron_index, (ref_times, cand_times) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        np.testing.assert_allclose(
            ref_times,
            cand_times,
            rtol=0.0,
            atol=atol_ms,
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
