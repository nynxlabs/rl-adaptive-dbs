"""Shared biophysical plant (Kumaravelu et al., 2016).

The native Python port of the Kumaravelu et al. (2016) model (``PythonPlant``,
the network integrator, the spectrum behind ``p_beta``) lives in the private
``kumaravelu_plant`` package while the model's license is pending; see
LICENSING.md. This package imports without it, and its names resolve to the
private package on first use (``envs.plant._kumaravelu``).
"""

from envs.plant._kumaravelu import PlantUnavailableError, deferred
from envs.plant.biomarkers import (
    MEHREGAN_BETA_BAND_HZ,
    SpectrumParams,
    band_power,
    multitaper_psd_point_process,
    neuron_band_power,
    p_beta,
)
from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec
from envs.plant.dbs import create_dbs_current
from envs.plant.matlab_backend import IntegrateResult, MatlabPlant

PythonPlant = deferred("python_backend", "PythonPlant")

__all__ = [
    "DbsSpec",
    "IntegrateResult",
    "MEHREGAN_BETA_BAND_HZ",
    "MatlabPlant",
    "PlantUnavailableError",
    "PythonPlant",
    "PlantConfig",
    "create_dbs_current",
    "SpectrumParams",
    "band_power",
    "multitaper_psd_point_process",
    "neuron_band_power",
    "p_beta",
]
