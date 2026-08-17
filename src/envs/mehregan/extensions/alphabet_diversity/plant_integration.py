"""Stitched idbs helper (unit tests). Env continuous mode uses 2 s plant carry."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from envs.mehregan.extensions.alphabet_diversity.observations import clip_spikes_to_subwindow
from envs.plant.biomarkers import p_beta
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.matlab_backend import IntegrateResult


class IdbsAlphabet(Protocol):
    def idbs_for_action(self, action: int) -> np.ndarray: ...


def resolve_mean_hz(alphabet: Any, *, fallback_hz: float) -> float:
    mean_hz = getattr(alphabet, "mean_hz", None)
    if mean_hz is not None:
        return float(mean_hz)
    return float(fallback_hz)


def idbs_segment_for_action(
    alphabet: Any,
    action: int,
    *,
    step_duration_s: float,
    dt_ms: float,
) -> np.ndarray:
    """Return one RL-step STN drive trace on the plant grid."""
    if hasattr(alphabet, "idbs_for_action"):
        return np.asarray(alphabet.idbs_for_action(int(action)), dtype=np.float64)
    spec = alphabet.to_dbs_spec(int(action))
    if spec.idbs is not None:
        return np.asarray(spec.idbs, dtype=np.float64)
    return create_dbs_current(
        spec.frequency_hz,
        tmax_ms=step_duration_s * 1000.0,
        dt_ms=dt_ms,
    )


def stitch_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int],
    alphabet: Any,
    rl_step_s: float,
) -> np.ndarray:
    """Stitch per-step idbs segments into one plant timeline (zeros before onset)."""
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    trace = np.zeros(n_steps, dtype=np.float64)
    onset_idx = int(round(onset_sim_s * 1000.0 / dt_ms))
    step_samples = int(round(rl_step_s * 1000.0 / dt_ms))
    for seg_i, action in enumerate(segment_actions):
        seg = idbs_segment_for_action(
            alphabet,
            int(action),
            step_duration_s=rl_step_s,
            dt_ms=dt_ms,
        )
        start = onset_idx + seg_i * step_samples
        end = min(start + seg.size, n_steps)
        if start >= n_steps:
            break
        trace[start:end] = seg[: end - start]
    return trace


def integrate_stitched_step(
    plant: Any,
    *,
    seed: int | None,
    pre_stim_s: float,
    step_duration_s: float,
    actions: list[int],
    alphabet: Any,
    dt_ms: float,
    mean_hz: float,
) -> IntegrateResult:
    """One continuous integrate from episode IC; return view of the latest RL step.

    Kept for unit tests of ``stitch_idbs`` placement. Env ``continuous`` mode
    uses sequential 2 s ``PythonPlant.integrate(..., carry=True)`` instead.
    """
    if not actions:
        msg = "integrate_stitched_step requires at least one action"
        raise ValueError(msg)

    total_s = pre_stim_s + len(actions) * step_duration_s
    idbs = stitch_idbs(
        duration_s=total_s,
        dt_ms=dt_ms,
        onset_sim_s=pre_stim_s,
        segment_actions=actions,
        alphabet=alphabet,
        rl_step_s=step_duration_s,
    )
    spec = DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(mean_hz).pick_dbs_freq,
        idbs=idbs,
        mean_hz=mean_hz,
    )
    plant.reset(seed)
    full = plant.integrate(total_s, spec, record_spikes=True)
    if not full.gpi_spikes:
        msg = "continuous integrate missing gpi_spikes"
        raise RuntimeError(msg)

    step_index = len(actions) - 1
    t0 = pre_stim_s + step_index * step_duration_s
    t1 = t0 + step_duration_s
    window_spikes = clip_spikes_to_subwindow(
        full.gpi_spikes, t_start_s=t0, t_end_s=t1
    )
    window_p_beta = float(
        p_beta(
            window_spikes,
            dt_ms=dt_ms,
            segment_duration_s=step_duration_s,
        )
    )
    return IntegrateResult(
        gpi_spikes=window_spikes,
        duration_s=step_duration_s,
        dt_ms=dt_ms,
        pd=full.pd,
        dbs_spec=alphabet.to_dbs_spec(int(actions[-1])),
        seed=seed,
        p_beta=window_p_beta,
        info={
            **(full.info or {}),
            "plant_integration_mode": "continuous",
            "continuous_total_s": total_s,
            "continuous_step_index": step_index,
            "continuous_window_t0_s": t0,
            "continuous_window_t1_s": t1,
        },
    )
