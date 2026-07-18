"""Plant configuration (Kumaravelu et al., 2016 reference defaults)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SmcSchedule = Literal["off", "boc", "periodic"]
SmcSite = Literal["thalamic", "cortical"]
SmcPulseSource = Literal["drive", "cor_spikes"]

# BoC platform defaults (Jovanov / Gao EI pulse amp & width; ~14 Hz inverse-gamma timing).
BOC_SMC_AMPLITUDE: float = 3.5
BOC_SMC_CORTICAL_AMPLITUDE: float = 50.0
BOC_SMC_PULSE_WIDTH_MS: float = 5.0
BOC_SMC_INVGAMMA_SHAPE: float = 25.0
BOC_SMC_INVGAMMA_SCALE_MS: float = 1785.71
# Kumaravelu TH defaults. So et al. (2012) drove TH with SMC pulses and no
# constant cerebellar bias; Fig 2b path A sets iappth_baseline=0 for that drive.
KUMARAVELU_IAPPTH_BASELINE: float = 1.2
KUMARAVELU_GGITH: float = 0.112


@dataclass(frozen=True)
class PlantConfig:
    """Non-Gym plant settings per docs/plant.md."""

    dt_ms: float = 0.01
    pd: int = 1
    corstim: int = 0
    neurons_per_region: int = 10
    # SMC for Error Index: thalamic Iappth (Gao/So) or cortical Iappco (probe path).
    smc_schedule: SmcSchedule = "off"
    smc_site: SmcSite = "thalamic"
    smc_pulse_source: SmcPulseSource = "drive"
    smc_frequency_hz: float = 0.0  # periodic schedule only
    smc_amplitude: float = BOC_SMC_AMPLITUDE
    smc_cortical_amplitude: float = BOC_SMC_CORTICAL_AMPLITUDE
    smc_pulse_width_ms: float = BOC_SMC_PULSE_WIDTH_MS
    smc_invgamma_shape: float = BOC_SMC_INVGAMMA_SHAPE
    smc_invgamma_scale_ms: float = BOC_SMC_INVGAMMA_SCALE_MS
    # TH bias / GPi→TH. Fig 2b: iappth_baseline=0 (So-style pulses-only).
    iappth_baseline: float = KUMARAVELU_IAPPTH_BASELINE
    ggith: float = KUMARAVELU_GGITH

    def smc_enabled(self) -> bool:
        if self.smc_schedule in ("boc", "periodic"):
            return True
        return self.smc_frequency_hz > 0.0
