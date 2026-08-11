"""Nguyen reward (Eq. (7)) and GPi α–β feedback."""

from __future__ import annotations

import numpy as np

from controllers.snn.config import SNNConfig
from controllers.snn.energy import dbs_energy_index
from envs.plant.biomarkers import REFERENCE_ALPHA_BETA_BAND_HZ, p_beta


def alpha_beta_power(
    gpi_spikes: list[np.ndarray],
    *,
    duration_s: float,
    dt_ms: float = 0.01,
) -> float:
    """GPi α–β oscillation power (7–35 Hz) for Nguyen feedback."""
    return p_beta(
        gpi_spikes,
        dt_ms=dt_ms,
        segment_duration_s=duration_s,
        f_low=REFERENCE_ALPHA_BETA_BAND_HZ[0],
        f_high=REFERENCE_ALPHA_BETA_BAND_HZ[1],
    )


def nguyen_reward(
    *,
    alpha_beta: float,
    energy: float,
    terminated: bool,
    remaining_steps: int,
    config: SNNConfig | None = None,
    prev_alpha_beta: float | None = None,
) -> float:
    """Nguyen Eq. (7) reward for one RL transition."""
    cfg = (config or SNNConfig()).with_variant_defaults()
    theta = cfg.alpha_beta_threshold
    theta_u = 1.0 if alpha_beta < theta else 0.0
    # Paper Eq. (7): raw squared distance to θ (no normalization); α–β is O(10²).
    d = float((alpha_beta - theta) ** 2) if theta_u == 0.0 else 0.0

    progress = 0.0
    if (
        theta_u == 0.0
        and prev_alpha_beta is not None
        and cfg.alpha_beta_progress_coef > 0.0
    ):
        progress = cfg.alpha_beta_progress_coef * max(0.0, float(prev_alpha_beta) - alpha_beta)
        if cfg.alpha_beta_progress_cap_per_step > 0.0:
            progress = min(progress, cfg.alpha_beta_progress_cap_per_step)

    warm = 0.0
    if (
        theta_u == 0.0
        and cfg.warm_zone_upper > theta
        and cfg.warm_zone_bonus_coef > 0.0
        and alpha_beta < cfg.warm_zone_upper
    ):
        warm = cfg.warm_zone_bonus_coef * (cfg.warm_zone_upper - alpha_beta)

    if terminated:
        return cfg.threshold_reward * (remaining_steps + 1) - cfg.energy_penalty * energy
    # Above threshold: penalize squared distance; below: τ bonus per step.
    return (
        -cfg.energy_penalty * energy
        + cfg.threshold_reward * theta_u
        - (1.0 - theta_u) * d
        + progress
        + warm
    )
