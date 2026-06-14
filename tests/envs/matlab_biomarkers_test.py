"""$P_\\beta$ equivalence vs MATLAB Chronux path (needs Signal Processing Toolbox)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant import DbsSpec, MatlabPlant, p_beta
from envs.plant.biomarkers import MEHREGAN_BETA_BAND_HZ


def _matlab_has_dpss(engine) -> bool:
    try:
        engine.eval("assert(exist('dpss','file')==2)", nargout=0)
    except Exception:
        return False
    return True


def _spikes_to_matlab_cell(engine, gpi_spikes: list) -> object:
    import matlab

    n = len(gpi_spikes)
    engine.eval(f"gpi_cell = cell(1,{n});", nargout=0)
    for index, spikes in enumerate(gpi_spikes, start=1):
        column = matlab.double(np.asarray(spikes, dtype=float).reshape(-1, 1).tolist())
        engine.workspace["tmp_spikes"] = column
        engine.eval(f"gpi_cell{{{index}}} = tmp_spikes;", nargout=0)
    return engine.workspace["gpi_cell"]


@pytest.fixture(scope="module")
def matlab_plant() -> MatlabPlant:
    with MatlabPlant() as plant:
        yield plant


@pytest.mark.matlab
def test_p_beta_matches_matlab_reference(matlab_plant: MatlabPlant) -> None:
    eng = matlab_plant._get_engine()
    if not _matlab_has_dpss(eng):
        pytest.skip("MATLAB Signal Processing Toolbox (dpss) not available")

    result = matlab_plant.reset(seed=42).integrate(2.0, DbsSpec.none())
    assert result.p_beta is not None

    py_val = p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        segment_duration_s=result.duration_s,
    )
    assert result.p_beta == pytest.approx(py_val)

    f_low, f_high = MEHREGAN_BETA_BAND_HZ
    matlab_cell = _spikes_to_matlab_cell(eng, result.gpi_spikes)
    matlab_val = eng.plant_p_beta(
        matlab_cell,
        float(result.dt_ms),
        float(f_low),
        float(f_high),
        nargout=1,
    )
    assert float(matlab_val) == pytest.approx(py_val, rel=0.05)


@pytest.mark.matlab
def test_integrate_populates_p_beta(matlab_plant: MatlabPlant) -> None:
    result = matlab_plant.reset(seed=7).integrate(2.0, DbsSpec.none())
    assert result.p_beta is not None
    assert result.p_beta > 0.0
    assert result.info["p_beta"] == pytest.approx(result.p_beta)
