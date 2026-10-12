"""Fig 5/6-style stitched Kumaravelu integration for Ravivarapu binary actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.sea_dbs.config import SEADBSConfig
from envs.mehregan.extensions.alphabet_diversity.observations import clip_spikes_to_subwindow
from envs.plant.biomarkers import p_beta
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.matlab_backend import IntegrateResult


def idbs_segment_for_action(
    action: int,
    *,
    duration_s: float,
    carrier_hz: float,
    burst_ms: float | None,
    dt_ms: float,
) -> np.ndarray:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    if int(action) == 0:
        return np.zeros(n_steps, dtype=np.float64)
    integration_ms = duration_s * 1000.0
    full = create_dbs_current(carrier_hz, tmax_ms=integration_ms, dt_ms=dt_ms)
    if burst_ms is not None and burst_ms < integration_ms:
        full[int(round(burst_ms / dt_ms)) :] = 0.0
    return full


def segment_schedule(actions: list[int], *, window_s: float) -> list[tuple[float, float, int]]:
    """Reset untreated onset + one equal-length entry per policy action."""
    segments: list[tuple[float, float, int]] = [(0.0, float(window_s), 0)]
    t = float(window_s)
    for action in actions:
        segments.append((t, t + float(window_s), int(action)))
        t += float(window_s)
    return segments


def stitch_binary_idbs(
    actions: list[int],
    *,
    window_s: float,
    carrier_hz: float,
    burst_ms: float | None,
    dt_ms: float,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    segments = segment_schedule(actions, window_s=window_s)
    parts = [
        idbs_segment_for_action(
            action,
            duration_s=t1 - t0,
            carrier_hz=carrier_hz,
            burst_ms=burst_ms,
            dt_ms=dt_ms,
        )
        for t0, t1, action in segments
    ]
    idbs = parts[0]
    for part in parts[1:]:
        idbs = np.concatenate([idbs, part[1:]])
    return idbs, segments


def integrate_stitched_binary_step(
    plant: Any,
    *,
    seed: int | None,
    config: SEADBSConfig,
    carrier_hz: float,
    actions: list[int],
) -> IntegrateResult:
    """One continuous integrate from episode IC; Pβ for the latest policy segment."""
    if not actions:
        msg = "integrate_stitched_binary_step requires at least one action"
        raise ValueError(msg)

    window_s = float(config.integration_duration_s)
    dt_ms = float(config.plant_dt_ms)
    idbs, segments = stitch_binary_idbs(
        actions,
        window_s=window_s,
        carrier_hz=float(carrier_hz),
        burst_ms=config.dbs_burst_ms,
        dt_ms=dt_ms,
    )
    total_s = segments[-1][1]
    spec = DbsSpec(pick_dbs_freq=2, idbs=idbs)
    plant.reset(seed)
    full = plant.integrate(total_s, spec, record_spikes=True)
    if not full.gpi_spikes:
        msg = "continuous integrate missing gpi_spikes"
        raise RuntimeError(msg)

    t0, t1, action = segments[-1]
    window_spikes = clip_spikes_to_subwindow(
        full.gpi_spikes, t_start_s=t0, t_end_s=t1
    )
    window_p_beta = float(
        p_beta(
            window_spikes,
            dt_ms=dt_ms,
            segment_duration_s=t1 - t0,
        )
    )
    return IntegrateResult(
        gpi_spikes=window_spikes,
        duration_s=t1 - t0,
        dt_ms=dt_ms,
        pd=full.pd,
        dbs_spec=DbsSpec.none() if int(action) == 0 else DbsSpec.from_frequency_hz(carrier_hz),
        seed=seed,
        p_beta=window_p_beta,
        info={
            **(full.info or {}),
            "plant_integration_mode": "continuous",
            "continuous_total_s": total_s,
            "continuous_step_index": len(actions) - 1,
            "continuous_window_t0_s": t0,
            "continuous_window_t1_s": t1,
        },
    )
