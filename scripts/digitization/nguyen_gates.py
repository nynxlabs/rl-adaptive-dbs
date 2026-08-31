"""Digitization-anchored gate helpers for Nguyen (paper 2) panels.

Loads normalized ``curves_fig*.json`` under
``artifacts/figures/papers/nguyen/paper_digitization/`` and compares
replication traces with **x-axis windows** (episode index, RL step).

Fig 3 is a scatter/boxplot — no refined time-series digitization yet; gates use
documented paper readouts (~215 / ~295 means) until ``curves_fig3`` exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from paper_gates import (
    DEFAULT_RATIO_TOL,
    DEFAULT_REL_TOL,
    _gate_pack,
    load_refined,
    pearson_on_ref_x,
    ratio_close,
    rel_close,
    window_mean,
)

ARTIFACT_ROOT = Path("artifacts/figures/papers/nguyen/paper_digitization")

# Paper Fig 3 qualitative readouts (tracker / panel note; not digitized curves).
PAPER_FIG3_PD_OFF_MEAN = 215.0
PAPER_FIG3_PD_ON_MEAN = 295.0
THETA = 150.0

INIT_AMP = 300.0
INIT_FREQ = 40.0
INIT_PW = 0.3


def curves_path(stem: str) -> Path:
    """Canonical normalized curves JSON for a Nguyen sub-panel."""
    return ARTIFACT_ROOT / f"curves_{stem}.json"


def load_curves(stem: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from ``curves_{stem}.json``."""
    return load_refined(curves_path(stem))


def _pick_series(
    paper: dict[str, tuple[np.ndarray, np.ndarray]],
    *aliases: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name in aliases:
        if name in paper:
            return paper[name]
    msg = f"none of {aliases!r} in paper series {list(paper)}"
    raise KeyError(msg)


def attach_digitization(
    heuristic: dict[str, Any],
    dig_report: dict[str, Any],
    *,
    prefix: str = "paper_",
) -> dict[str, Any]:
    """Merge heuristic gate dict with a digitization report; ``pass`` = both."""
    out = dict(heuristic)
    for key, value in dig_report.get("gates", {}).items():
        out[f"{prefix}{key}"] = bool(value)
    out["paper_gate_metrics"] = dig_report.get("metrics", {})
    out["paper_ref"] = dig_report.get("paper_ref", {})
    out["paper_notes"] = list(dig_report.get("notes", []))
    h_pass = bool(heuristic.get("pass", False))
    d_pass = bool(dig_report.get("pass", True))
    out["pass"] = h_pass and d_pass
    return out


def fig3_gates(
    samples: dict[str, Any],
    *,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 3 distribution: ordering + mean ratio vs documented paper readouts."""
    pd_off = np.asarray(samples["pd_off"], dtype=float)
    pd_on = np.asarray(samples["pd_on"], dtype=float)
    mean_off = float(np.mean(pd_off))
    mean_on = float(np.mean(pd_on))
    median_off = float(np.median(pd_off))
    median_on = float(np.median(pd_on))
    pd_q1 = float(np.percentile(pd_on, 25))

    gates = {
        "ordering_pd_on_above_pd_off": median_on > median_off,
        "mean_ratio_near_paper_readout": ratio_close(
            mean_on,
            mean_off,
            PAPER_FIG3_PD_ON_MEAN,
            PAPER_FIG3_PD_OFF_MEAN,
            tol=rel_tol,
        ),
        "means_separated": mean_on > mean_off + 0.15 * max(mean_off, 1.0),
    }
    return _gate_pack(
        gates,
        {
            "pd_off_mean": mean_off,
            "pd_on_mean": mean_on,
            "pd_off_median": median_off,
            "pd_on_median": median_on,
            "pd_on_q1": pd_q1,
            "threshold_near_pd_on_q1": bool(abs(pd_q1 - THETA) / max(THETA, 1.0) < 0.75),
            "paper_pd_off_mean": PAPER_FIG3_PD_OFF_MEAN,
            "paper_pd_on_mean": PAPER_FIG3_PD_ON_MEAN,
        },
        paper_ref={
            "path": None,
            "note": "Fig 3 scatter/boxplot — no curves_fig3.json; using tracker readouts",
        },
        notes=[
            "threshold_near_pd_on_q1 is soft/informational.",
            "Digitize Fig 3 panel (a) to replace mean_ratio_near_paper_readout anchor.",
        ],
    )


def fig4_reward_gates(
    episode_rewards: list[float] | np.ndarray,
    *,
    early_hi: float = 50.0,
    late_lo: float = 350.0,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 4 panel (a) reward vs digitized paper training curve."""
    rewards = np.asarray(episode_rewards, dtype=float)
    n = int(rewards.size)
    if n < 10:
        return _gate_pack(
            {"enough_episodes": False},
            {"n_episodes": n},
            paper_ref={"reward": str(curves_path("fig4_reward")), "early_hi": early_hi, "late_lo": late_lo},
        )

    x = np.arange(n, dtype=float)
    late_r = window_mean(x, rewards, lo=late_lo)
    first50_r = window_mean(x, rewards, hi=min(50.0, float(n - 1)))

    paper_r = load_curves("fig4_reward")
    prx, pry = _pick_series(paper_r, "Smoothed", "Raw")
    p_late_r = window_mean(prx, pry, lo=late_lo)
    p_first50_r = window_mean(prx, pry, hi=50.0)

    # Ship dig gate: direction only. Magnitude / late-early ratio vs paper assume
    # the paper's negative-million reward band; locked progress shaping yields
    # positive rewards, so those checks are diagnostic (logged, not required).
    gates = {
        "reward_improves_like_paper": bool(
            late_r > first50_r and p_late_r > p_first50_r
        ),
    }
    diagnostics = {
        "early_reward_mag_near_paper": rel_close(
            abs(first50_r), abs(p_first50_r), tol=rel_tol
        ),
        "late_reward_ratio_near_paper": ratio_close(
            late_r, first50_r, p_late_r, p_first50_r, tol=ratio_tol
        ),
    }
    shape_r = pearson_on_ref_x(prx, pry, x, rewards)

    pack = _gate_pack(
        gates,
        {
            "early_reward": first50_r,
            "late_reward": late_r,
            "paper_early_reward": p_first50_r,
            "paper_late_reward": p_late_r,
            "pearson_reward": shape_r,
            "early_reward_mag_near_paper": diagnostics["early_reward_mag_near_paper"],
            "late_reward_ratio_near_paper": diagnostics["late_reward_ratio_near_paper"],
        },
        paper_ref={
            "reward": str(curves_path("fig4_reward")),
            "early_hi": early_hi,
            "late_lo": late_lo,
        },
        notes=[
            "Pearson r is diagnostic only; seeds change wiggles.",
            "early_reward_mag_near_paper and late_reward_ratio_near_paper are "
            "diagnostic under positive reward shaping (paper band is negative-million).",
        ],
    )
    pack["gates"].update(diagnostics)
    return pack


def fig4_length_gates(
    episode_lengths: list[int] | np.ndarray,
    *,
    max_episode_steps: int = 25,
    late_lo: float = 350.0,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 4 panel (b) episode length vs digitized paper training curve."""
    lengths = np.asarray(episode_lengths, dtype=float)
    n = int(lengths.size)
    if n < 10:
        return _gate_pack(
            {"enough_episodes": False},
            {"n_episodes": n},
            paper_ref={"length": str(curves_path("fig4_length")), "late_lo": late_lo},
        )

    x = np.arange(n, dtype=float)
    early_len = window_mean(x, lengths, hi=75.0)
    late_len = window_mean(x, lengths, lo=late_lo)

    paper_l = load_curves("fig4_length")
    plx, ply = _pick_series(paper_l, "Smoothed", "Raw")
    p_early_len = window_mean(plx, ply, hi=75.0)
    p_late_len = window_mean(plx, ply, lo=late_lo)

    gates = {
        "length_decreases_like_paper": bool(late_len < early_len and p_late_len < p_early_len),
        "late_length_near_paper": rel_close(late_len, p_late_len, tol=rel_tol),
        "early_near_max_length": float(np.median(lengths[: min(50, n)])) >= max_episode_steps - 2,
    }
    shape_l = pearson_on_ref_x(plx, ply, x, lengths.astype(float))

    return _gate_pack(
        gates,
        {
            "early_length": early_len,
            "late_length": late_len,
            "paper_early_length": p_early_len,
            "paper_late_length": p_late_len,
            "pearson_length": shape_l,
        },
        paper_ref={
            "length": str(curves_path("fig4_length")),
            "late_lo": late_lo,
        },
        notes=["Pearson r is diagnostic only; seeds change wiggles."],
    )


# Fig 4 training-curve timing (explore wiggle → mid glide → post-100 plateau).
FIG4_TIMING_SMOOTH = 20
FIG4_MID_GLIDE = (50.0, 100.0)
FIG4_BY_100 = (80.0, 100.0)
FIG4_EARLY_SMOOTHED_MIN = 23.0
FIG4_REWARD_BY_100_FLOOR = -2.0e5
FIG4_POST_PLATEAU_LEVEL = (100.0, 150.0)
FIG4_POST_PLATEAU_SLOPE = (100.0, 250.0)
FIG4_POST_PLATEAU_RANGE = (100.0, 200.0)
FIG4_REWARD_PLATEAU_LEVEL = (175.0, 325.0)
FIG4_REWARD_PLATEAU_RANGE = (100.0, 250.0)
FIG4_REWARD_PLATEAU_SLOPE = (100.0, 450.0)
FIG4_TIMING_REL_TOL = 0.35
FIG4_LATE_LEVEL = (350.0, 500.0)
FIG4_LATE_SLOPE = (350.0, 490.0)
FIG4_LATE_LENGTH_LEVEL_MAX = 14.0
FIG4_LATE_LENGTH_SLOPE_MAX = 0.02
FIG4_LATE_TIMEOUT_MAX = 0.25


def _episode_smoothed(
    y: np.ndarray | list[float],
    *,
    smooth_window: int = FIG4_TIMING_SMOOTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Episode-indexed moving average (matches panel ``smooth_window`` default)."""
    arr = np.asarray(y, dtype=float)
    n = int(arr.size)
    if n < smooth_window:
        return np.arange(n, dtype=float), arr
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    sm = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(smooth_window - 1, n, dtype=float)
    return xs, sm


def _median_in_window(xs: np.ndarray, ys: np.ndarray, lo: float, hi: float) -> float:
    mask = (xs >= lo) & (xs < hi)
    if not np.any(mask):
        return float("nan")
    return float(np.median(ys[mask]))


def _ptp_in_window(xs: np.ndarray, ys: np.ndarray, lo: float, hi: float) -> float:
    mask = (xs >= lo) & (xs < hi)
    if not np.any(mask):
        return float("nan")
    return float(np.ptp(ys[mask]))


def _slope_in_window(xs: np.ndarray, ys: np.ndarray, lo: float, hi: float) -> float:
    mask = (xs >= lo) & (xs < hi)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.polyfit(xs[mask], ys[mask], 1)[0])


def fig4_timing_shape_gates(
    episode_lengths: list[int] | np.ndarray,
    episode_rewards: list[float] | np.ndarray,
    *,
    smooth_window: int = FIG4_TIMING_SMOOTH,
    rel_tol: float = FIG4_TIMING_REL_TOL,
    max_episode_steps: int = 25,
) -> dict[str, Any]:
    """Mid glide (ep 50–100) and post-100 plateau vs digitized paper smoothed curves."""
    lengths = np.asarray(episode_lengths, dtype=float)
    rewards = np.asarray(episode_rewards, dtype=float)
    n = int(lengths.size)
    if n < 100:
        return {
            "length_gates": {
                "length_early_smoothed_near_horizon": False,
                "length_mid_glide_like_paper": False,
                "length_by_100_near_paper": False,
                "length_post100_plateau": False,
                "late_length_no_regression": False,
                "late_timeout_fraction": False,
                "late_length_level": False,
            },
            "reward_gates": {
                "reward_improves_by_100": False,
                "reward_by_100_near_zero": False,
                "reward_post100_plateau": False,
            },
            "metrics": {"n_episodes": n, "reason": "too_few_episodes_for_timing"},
        }

    lx, ls = _episode_smoothed(lengths, smooth_window=smooth_window)
    rx, rs = _episode_smoothed(rewards, smooth_window=smooth_window)

    paper_l = load_curves("fig4_length")
    plx, ply = _pick_series(paper_l, "Smoothed", "Raw")
    paper_r = load_curves("fig4_reward")
    prx, pry = _pick_series(paper_r, "Smoothed", "Raw")

    mid_lo, mid_hi = FIG4_MID_GLIDE
    lvl_lo, lvl_hi = FIG4_POST_PLATEAU_LEVEL
    slope_lo, slope_hi = FIG4_POST_PLATEAU_SLOPE
    range_lo, range_hi = FIG4_POST_PLATEAU_RANGE

    len_early_0_50 = _median_in_window(lx, ls, 0.0, 50.0)
    len_mid_50_100 = _median_in_window(lx, ls, mid_lo, mid_hi)
    by100_lo, by100_hi = FIG4_BY_100
    len_80_100 = _median_in_window(lx, ls, by100_lo, by100_hi)
    len_lvl_100_150 = _median_in_window(lx, ls, lvl_lo, lvl_hi)
    len_slope_100_250 = _slope_in_window(lx, ls, slope_lo, slope_hi)
    len_ptp_100_200 = _ptp_in_window(lx, ls, range_lo, range_hi)

    p_len_early = _median_in_window(plx, ply, 0.0, 50.0)
    p_len_mid = _median_in_window(plx, ply, mid_lo, mid_hi)
    p_len_80_100 = _median_in_window(plx, ply, by100_lo, by100_hi)
    p_len_lvl = _median_in_window(plx, ply, lvl_lo, lvl_hi)

    length_early_smoothed = bool(len_early_0_50 >= FIG4_EARLY_SMOOTHED_MIN)
    length_mid_glide = bool(
        len_mid_50_100 < len_early_0_50 - 1.5
        and rel_close(len_mid_50_100, p_len_mid, tol=rel_tol)
    )
    length_by_100 = bool(rel_close(len_80_100, p_len_80_100, tol=rel_tol))
    length_post100 = bool(
        abs(len_slope_100_250) <= 0.055
        and len_ptp_100_200 <= 4.5
        and rel_close(len_lvl_100_150, p_len_lvl, tol=rel_tol)
    )

    rw_lo, rw_hi = FIG4_REWARD_PLATEAU_LEVEL
    rw_rng_lo, rw_rng_hi = FIG4_REWARD_PLATEAU_RANGE
    rw_slope_lo, rw_slope_hi = FIG4_REWARD_PLATEAU_SLOPE
    rew_med_late = _median_in_window(rx, rs, rw_lo, rw_hi)
    rew_0_50 = _median_in_window(rx, rs, 0.0, 50.0)
    rew_80_100 = _median_in_window(rx, rs, by100_lo, by100_hi)
    rew_ptp_100_250 = _ptp_in_window(rx, rs, rw_rng_lo, rw_rng_hi)
    rew_slope_100_450 = _slope_in_window(rx, rs, rw_slope_lo, rw_slope_hi)
    scale = max(abs(rew_med_late), 1.0e5)
    reward_post100 = bool(
        rew_ptp_100_250 <= max(0.30 * scale, 5.0e4)
        and abs(rew_slope_100_450) <= max(0.00015 * scale, 150.0)
    )
    reward_improves_by_100 = bool(rew_80_100 > rew_0_50)
    reward_by_100_near_zero = bool(rew_80_100 > FIG4_REWARD_BY_100_FLOOR)

    late_lo, late_hi = FIG4_LATE_LEVEL
    slope_lo, slope_hi = FIG4_LATE_SLOPE
    late_len_mean = _median_in_window(lx, ls, late_lo, late_hi)
    late_len_slope = _slope_in_window(lx, ls, slope_lo, slope_hi)
    late_start = int(late_lo)
    late_end = min(int(late_hi), n)
    raw_late = lengths[late_start:late_end]
    late_timeout_rate = (
        float(np.mean(raw_late >= float(max_episode_steps) - 0.5)) if raw_late.size else 1.0
    )
    late_length_level = bool(
        np.isfinite(late_len_mean) and late_len_mean <= FIG4_LATE_LENGTH_LEVEL_MAX
    )
    late_length_no_regression = bool(
        np.isfinite(late_len_slope) and late_len_slope <= FIG4_LATE_LENGTH_SLOPE_MAX
    )
    late_timeout_fraction = bool(late_timeout_rate <= FIG4_LATE_TIMEOUT_MAX)

    return {
        "length_gates": {
            "length_early_smoothed_near_horizon": length_early_smoothed,
            "length_mid_glide_like_paper": length_mid_glide,
            "length_by_100_near_paper": length_by_100,
            "length_post100_plateau": length_post100,
            "late_length_no_regression": late_length_no_regression,
            "late_timeout_fraction": late_timeout_fraction,
            "late_length_level": late_length_level,
        },
        "reward_gates": {
            "reward_improves_by_100": reward_improves_by_100,
            "reward_by_100_near_zero": reward_by_100_near_zero,
            "reward_post100_plateau": reward_post100,
        },
        "metrics": {
            "len_early_0_50": len_early_0_50,
            "len_mid_50_100": len_mid_50_100,
            "len_80_100": len_80_100,
            "len_lvl_100_150": len_lvl_100_150,
            "len_slope_100_250": len_slope_100_250,
            "len_ptp_100_200": len_ptp_100_200,
            "paper_len_early_0_50": p_len_early,
            "paper_len_mid_50_100": p_len_mid,
            "paper_len_80_100": p_len_80_100,
            "paper_len_lvl_100_150": p_len_lvl,
            "rew_0_50": rew_0_50,
            "rew_80_100": rew_80_100,
            "rew_med_175_325": rew_med_late,
            "rew_ptp_100_250": rew_ptp_100_250,
            "rew_slope_100_450": rew_slope_100_450,
            "late_len_mean_350_500": late_len_mean,
            "late_len_slope_350_490": late_len_slope,
            "late_timeout_rate_350_500": late_timeout_rate,
        },
    }


def fig4_training_gates(
    episode_rewards: list[float] | np.ndarray,
    episode_lengths: list[int] | np.ndarray,
    *,
    max_episode_steps: int = 25,
    early_hi: float = 50.0,
    late_lo: float = 350.0,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 4 reward + length vs digitized paper training curves (grouped)."""
    reward = fig4_reward_gates(
        episode_rewards,
        early_hi=early_hi,
        late_lo=late_lo,
        ratio_tol=ratio_tol,
        rel_tol=rel_tol,
    )
    length = fig4_length_gates(
        episode_lengths,
        max_episode_steps=max_episode_steps,
        late_lo=late_lo,
        rel_tol=rel_tol,
    )
    reward_gates = {f"paper_{k}": v for k, v in reward["gates"].items()}
    length_gates = {f"paper_{k}": v for k, v in length["gates"].items()}
    metrics = {
        **reward.get("metrics", {}),
        **length.get("metrics", {}),
    }
    return {
        "reward": reward,
        "length": length,
        "pass": bool(reward["pass"] and length["pass"]),
        "gates": {**reward_gates, **length_gates},
        "metrics": metrics,
        "paper_ref": {
            "reward": str(curves_path("fig4_reward")),
            "length": str(curves_path("fig4_length")),
            "early_hi": early_hi,
            "late_lo": late_lo,
        },
        "notes": list(reward.get("notes", [])),
    }


def fig5_spikes_energy_gates(
    episode_spikes: list[float] | np.ndarray,
    episode_energies: list[float] | np.ndarray,
    *,
    early_hi: float = 50.0,
    mid_lo: float = 55.0,
    mid_hi: float = 75.0,
    late_lo: float = 350.0,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
) -> dict[str, Any]:
    """Fig 5 spike count + DBS energy vs digitized paper curves."""
    spikes = np.asarray(episode_spikes, dtype=float)
    energies = np.asarray(episode_energies, dtype=float)
    n = int(spikes.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})

    x = np.arange(n, dtype=float)
    spike_mean = float(np.mean(spikes))
    energy_mean = float(np.mean(energies))
    spike_early = window_mean(x, spikes, hi=early_hi)
    spike_mid = window_mean(x, spikes, lo=mid_lo, hi=mid_hi)
    spike_late = window_mean(x, spikes, lo=late_lo)
    energy_early = window_mean(x, energies, hi=early_hi)
    energy_mid = window_mean(x, energies, lo=mid_lo, hi=mid_hi)
    energy_late = window_mean(x, energies, lo=late_lo)

    paper_s = load_curves("fig5_spikes")
    paper_e = load_curves("fig5_energy")
    psx, psy = _pick_series(paper_s, "Spike Count", "Smoothed", "Raw")
    pex, pey = _pick_series(paper_e, "Smoothed", "Raw")

    p_spike_mean = float(np.mean(psy))
    p_energy_mean = float(np.mean(pey))
    p_spike_early = window_mean(psx, psy, hi=early_hi)
    p_spike_mid = window_mean(psx, psy, lo=mid_lo, hi=mid_hi)
    p_spike_late = window_mean(psx, psy, lo=late_lo)
    p_energy_early = window_mean(pex, pey, hi=early_hi)
    p_energy_mid = window_mean(pex, pey, lo=mid_lo, hi=mid_hi)
    p_energy_late = window_mean(pex, pey, lo=late_lo)

    pearson_e = pearson_on_ref_x(pex, pey, x, energies)
    pearson_s = pearson_on_ref_x(psx, psy, x, spikes)

    gates = {
        "spike_early_near_paper": rel_close(spike_early, p_spike_early, tol=rel_tol),
        "spike_mid_near_paper": rel_close(spike_mid, p_spike_mid, tol=rel_tol),
        "spike_late_near_paper": rel_close(spike_late, p_spike_late, tol=rel_tol),
        "spike_mean_near_paper": rel_close(spike_mean, p_spike_mean, tol=rel_tol),
        "spike_trend_near_paper": ratio_close(
            spike_late, spike_early, p_spike_late, p_spike_early, tol=ratio_tol
        ),
        "spike_stays_near_800": bool(
            600.0 <= spike_early <= 1000.0
            and 600.0 <= spike_mid <= 1000.0
            and 600.0 <= spike_late <= 1000.0
            and 600.0 <= spike_mean <= 1000.0
        ),
        "energy_early_near_paper": rel_close(energy_early, p_energy_early, tol=rel_tol),
        "energy_mid_near_paper": rel_close(energy_mid, p_energy_mid, tol=rel_tol),
        "energy_late_near_paper": rel_close(energy_late, p_energy_late, tol=rel_tol),
        "energy_mean_near_paper": rel_close(energy_mean, p_energy_mean, tol=rel_tol),
        "energy_mid_ramp_near_paper": ratio_close(
            energy_mid, energy_early, p_energy_mid, p_energy_early, tol=ratio_tol
        ),
        "energy_trend_near_paper": ratio_close(
            energy_late, energy_early, p_energy_late, p_energy_early, tol=ratio_tol
        ),
        "energy_monotonic_rise": bool(energy_early < energy_mid < energy_late),
        "energy_late_above_early": bool(energy_late > energy_early * 1.25),
        "spike_series_has_variance": float(np.std(spikes)) > 0.0,
        "energy_not_constant": float(np.std(energies)) > 0.01 * max(abs(energy_mean), 1.0),
    }
    return _gate_pack(
        gates,
        {
            "spike_mean": spike_mean,
            "energy_mean": energy_mean,
            "spike_early": spike_early,
            "spike_mid": spike_mid,
            "spike_late": spike_late,
            "energy_early": energy_early,
            "energy_mid": energy_mid,
            "energy_late": energy_late,
            "paper_spike_mean": p_spike_mean,
            "paper_energy_mean": p_energy_mean,
            "paper_spike_early": p_spike_early,
            "paper_spike_mid": p_spike_mid,
            "paper_spike_late": p_spike_late,
            "paper_energy_early": p_energy_early,
            "paper_energy_mid": p_energy_mid,
            "paper_energy_late": p_energy_late,
            "pearson_energy": pearson_e,
            "pearson_spikes": pearson_s,
        },
        paper_ref={
            "spikes": str(curves_path("fig5_spikes")),
            "energy": str(curves_path("fig5_energy")),
        },
        notes=[
            "Spikes digitization: single traced series (Spike Count); no separate Raw export.",
            "Energy mid window ep 55–75 targets paper ramp (~ep 60–70); compare Smoothed curve.",
        ],
    )


def fig6_power_gates(
    episode_alpha_beta: list[float] | np.ndarray,
    *,
    early_hi: float = 50.0,
    mid_lo: float = 50.0,
    mid_hi: float = 100.0,
    post100_lo: float = 100.0,
    post100_hi: float = 250.0,
    late_lo: float = 350.0,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    theta: float = THETA,
) -> dict[str, Any]:
    """Fig 6 panel (a) GPi α–β oscillation power vs digitized paper curve."""
    ab = np.asarray(episode_alpha_beta, dtype=float)
    n = int(ab.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})

    x = np.arange(n, dtype=float)
    ab_early = window_mean(x, ab, hi=early_hi)
    ab_mid = window_mean(x, ab, lo=mid_lo, hi=mid_hi)
    ab_post100 = window_mean(x, ab, lo=post100_lo, hi=post100_hi)
    ab_late = window_mean(x, ab, lo=late_lo)
    ab_mean = float(np.mean(ab))
    ab_drop_ratio = ab_late / max(ab_early, 1e-9)
    ab_drop_mag = ab_early - ab_late

    paper_ab = load_curves("fig6_power")
    pabx, paby = _pick_series(paper_ab, "Smoothed", "Raw")

    p_ab_early = window_mean(pabx, paby, hi=early_hi)
    p_ab_mid = window_mean(pabx, paby, lo=mid_lo, hi=mid_hi)
    p_ab_post100 = window_mean(pabx, paby, lo=post100_lo, hi=post100_hi)
    p_ab_late = window_mean(pabx, paby, lo=late_lo)
    p_ab_mean = float(np.mean(paby))
    p_ab_drop_ratio = p_ab_late / max(p_ab_early, 1e-9)
    p_ab_drop_mag = p_ab_early - p_ab_late

    pearson_ab = pearson_on_ref_x(pabx, paby, x, ab)
    late_std = float(np.std(ab[int(late_lo):])) if n > int(late_lo) else float("nan")

    gates = {
        "alpha_beta_early_above_theta": bool(ab_early > theta),
        "alpha_beta_early_near_paper": rel_close(ab_early, p_ab_early, tol=rel_tol),
        "alpha_beta_mid_near_paper": rel_close(ab_mid, p_ab_mid, tol=rel_tol),
        "alpha_beta_drops_by_100": bool(ab_mid < ab_early - 15.0),
        "alpha_beta_post100_below_theta": bool(ab_post100 <= theta),
        "alpha_beta_post100_near_paper": rel_close(ab_post100, p_ab_post100, tol=rel_tol),
        "alpha_beta_late_below_theta": bool(ab_late <= theta),
        "alpha_beta_late_near_paper": rel_close(ab_late, p_ab_late, tol=rel_tol),
        "alpha_beta_mean_near_paper": rel_close(ab_mean, p_ab_mean, tol=rel_tol),
        "alpha_beta_monotonic_drop": bool(ab_early > ab_mid > ab_late),
        "alpha_beta_trend_near_paper": ratio_close(ab_late, ab_early, p_ab_late, p_ab_early, tol=ratio_tol),
        "alpha_beta_drop_magnitude_near_paper": rel_close(ab_drop_mag, p_ab_drop_mag, tol=rel_tol),
        "alpha_beta_series_has_variance": bool(float(np.std(ab)) > 0.0),
        "alpha_beta_late_stable": bool(late_std <= 0.35 * max(abs(ab_late), 1.0)),
        "alpha_beta_decreases_like_paper": bool(ab_late < ab_early and p_ab_late < p_ab_early),
    }
    metrics = {
        "alpha_beta_early": ab_early,
        "alpha_beta_mid": ab_mid,
        "alpha_beta_post100": ab_post100,
        "alpha_beta_late": ab_late,
        "alpha_beta_mean": ab_mean,
        "alpha_beta_drop_ratio": ab_drop_ratio,
        "alpha_beta_drop_magnitude": ab_drop_mag,
        "paper_alpha_beta_early": p_ab_early,
        "paper_alpha_beta_mid": p_ab_mid,
        "paper_alpha_beta_post100": p_ab_post100,
        "paper_alpha_beta_late": p_ab_late,
        "paper_alpha_beta_mean": p_ab_mean,
        "paper_alpha_beta_drop_ratio": p_ab_drop_ratio,
        "paper_alpha_beta_drop_magnitude": p_ab_drop_mag,
        "pearson_alpha_beta": pearson_ab,
    }
    return _gate_pack(
        gates,
        metrics,
        paper_ref={"power": str(curves_path("fig6_power"))},
        notes=["Fig 6a: GPi α–β oscillation power (7–35 Hz) over 500 training episodes."],
    )


def fig6_training_gates(
    episode_alpha_beta: list[float] | np.ndarray,
    episode_amplitudes: list[float] | np.ndarray,
    episode_frequencies: list[float] | np.ndarray,
    episode_pulse_widths: list[float] | np.ndarray,
    *,
    early_hi: float = 50.0,
    mid_lo: float = 50.0,
    mid_hi: float = 100.0,
    post100_lo: float = 100.0,
    post100_hi: float = 250.0,
    late_lo: float = 350.0,
    late_stable_n: int = 50,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
) -> dict[str, Any]:
    """Fig 6 α–β + DBS parameters over training vs digitized paper curves."""
    ab = np.asarray(episode_alpha_beta, dtype=float)
    amp = np.asarray(episode_amplitudes, dtype=float)
    freq = np.asarray(episode_frequencies, dtype=float)
    pw = np.asarray(episode_pulse_widths, dtype=float)
    n = int(ab.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})

    power_report = fig6_power_gates(
        ab,
        early_hi=early_hi,
        mid_lo=mid_lo,
        mid_hi=mid_hi,
        post100_lo=post100_lo,
        post100_hi=post100_hi,
        late_lo=late_lo,
        rel_tol=rel_tol,
        ratio_tol=ratio_tol,
    )

    x = np.arange(n, dtype=float)
    amp_late = window_mean(x, amp, lo=late_lo)
    freq_late = window_mean(x, freq, lo=late_lo)
    pw_late = window_mean(x, pw, lo=late_lo)
    stable_start = max(0, n - late_stable_n)
    amp_std_late = float(np.std(amp[stable_start:]))
    freq_std_late = float(np.std(freq[stable_start:]))
    pw_std_late = float(np.std(pw[stable_start:]))

    paper_amp = load_curves("fig6_amp")
    paper_freq = load_curves("fig6_freq")
    paper_pw = load_curves("fig6_pw")
    pampx, pampy = _pick_series(paper_amp, "Smoothed", "Raw")
    pfreqx, pfreqy = _pick_series(paper_freq, "Smoothed", "Raw")
    ppwx, ppwy = _pick_series(paper_pw, "Smoothed", "Raw")

    p_amp_late = window_mean(pampx, pampy, lo=late_lo)
    p_freq_late = window_mean(pfreqx, pfreqy, lo=late_lo)
    p_pw_late = window_mean(ppwx, ppwy, lo=late_lo)

    amp_left = abs(amp_late - INIT_AMP) / INIT_AMP > 0.05
    freq_left = abs(freq_late - INIT_FREQ) / INIT_FREQ > 0.05
    pw_left = abs(pw_late - INIT_PW) / max(INIT_PW, 1e-9) > 0.05

    param_gates = {
        "amp_left_init": amp_left,
        "freq_left_init": freq_left,
        "pw_left_init": pw_left,
        "params_left_init": bool(amp_left and freq_left and pw_left),
        "amp_late_near_paper": rel_close(amp_late, p_amp_late, tol=rel_tol),
        "freq_late_near_paper": rel_close(freq_late, p_freq_late, tol=rel_tol),
        "pw_late_near_paper": rel_close(pw_late, p_pw_late, tol=rel_tol),
        "late_params_stable": all(
            std <= 0.20 * max(abs(mean), 1e-9)
            for std, mean in (
                (amp_std_late, amp_late),
                (freq_std_late, freq_late),
                (pw_std_late, pw_late),
            )
        ),
    }

    all_gates = {**power_report["gates"], **param_gates}
    metrics = {
        **power_report["metrics"],
        "amp_late": amp_late,
        "freq_late": freq_late,
        "pw_late": pw_late,
        "paper_amp_late": p_amp_late,
        "paper_freq_late": p_freq_late,
        "paper_pw_late": p_pw_late,
    }
    return _gate_pack(
        all_gates,
        metrics,
        paper_ref={
            "power": str(curves_path("fig6_power")),
            "amp": str(curves_path("fig6_amp")),
            "freq": str(curves_path("fig6_freq")),
            "pw": str(curves_path("fig6_pw")),
        },
        notes=[
            "amp/freq/pw late anchors are soft; shape gates are primary.",
            "Fig 6a: GPi α–β oscillation power (7–35 Hz) over 500 training episodes.",
        ],
    )


def fig7_eval_gates(
    alpha_beta_trajectories: list[list[float]],
    *,
    fig3_pd_on_median: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
) -> dict[str, Any]:
    """Fig 7 eval mean α–β trace vs digitized paper average curve."""
    if not alpha_beta_trajectories:
        return _gate_pack({"trajectories_present": False}, {})

    # Per-step mean across episodes (paper shows mean over 50 episodes).
    max_len = max(len(tr) for tr in alpha_beta_trajectories)
    step_means: list[float] = []
    for step in range(max_len):
        vals = [float(tr[step]) for tr in alpha_beta_trajectories if step < len(tr)]
        step_means.append(float(np.mean(vals)) if vals else float("nan"))
    steps = np.arange(len(step_means), dtype=float)
    mean_trace = np.asarray(step_means, dtype=float)
    overall_mean = float(np.nanmean(mean_trace)) if mean_trace.size else float("nan")

    paper = load_curves("fig7")
    pax, pay = _pick_series(paper, "average", "Smoothed", "Raw")
    p_overall = float(np.mean(pay))
    p_start = float(pay[0])
    p_peak = float(np.max(pay))
    p_peak_x = float(pax[np.argmax(pay)])
    p_early = window_mean(pax, pay, hi=5.0)
    p_mid = window_mean(pax, pay, lo=6.0, hi=15.0)
    p_late = window_mean(pax, pay, lo=18.0)
    p_drop = p_peak - p_late
    p_ratio = p_late / max(p_peak, 1e-9)

    start_val = float(mean_trace[0]) if mean_trace.size else float("nan")
    peak_val = float(np.nanmax(mean_trace)) if mean_trace.size else float("nan")
    peak_step = int(np.nanargmax(mean_trace)) if mean_trace.size else -1
    early = window_mean(steps, mean_trace, hi=5.0)
    mid = window_mean(steps, mean_trace, lo=6.0, hi=15.0)
    late = window_mean(steps, mean_trace, lo=18.0)
    drop_val = peak_val - late
    late_peak_ratio = late / max(peak_val, 1e-9)
    r_shape = pearson_on_ref_x(pax, pay, steps, mean_trace)

    gates: dict[str, bool] = {
        "eval_protocol_ok": len(alpha_beta_trajectories) >= 1 and max_len >= 20,
        "step_series_finite": bool(np.isfinite(mean_trace).all()),
        "start_near_paper": rel_close(start_val, p_start, tol=rel_tol),
        "peak_step_timing": peak_step in (1, 2, 3),
        "peak_power_near_paper": rel_close(peak_val, p_peak, tol=rel_tol),
        "early_mean_near_paper": rel_close(early, p_early, tol=rel_tol),
        "mid_mean_near_paper": rel_close(mid, p_mid, tol=rel_tol),
        "late_mean_near_paper": rel_close(late, p_late, tol=rel_tol),
        "overall_mean_near_paper": rel_close(overall_mean, p_overall, tol=rel_tol),
        "peak_to_late_drop": bool(peak_val > late + 40.0 and rel_close(drop_val, p_drop, tol=0.50)),
        "late_peak_ratio_near_paper": ratio_close(late, peak_val, p_late, p_peak, tol=ratio_tol),
        "pearson_shape_ok": bool(np.isfinite(r_shape) and r_shape >= 0.80),
    }
    metrics_extra: dict[str, Any] = {
        "mean_below_theta": overall_mean <= THETA,
        "paper_mean_below_theta": p_overall <= THETA,
        "late_below_theta": late <= 165.0,
    }
    if fig3_pd_on_median is not None and np.isfinite(fig3_pd_on_median):
        gates["below_fig3_pd_median"] = overall_mean < float(fig3_pd_on_median)

    return _gate_pack(
        gates,
        {
            "start_val": start_val,
            "paper_start_val": p_start,
            "peak_val": peak_val,
            "paper_peak_val": p_peak,
            "peak_step": peak_step,
            "paper_peak_step": p_peak_x,
            "early_mean": early,
            "paper_early_mean": p_early,
            "mid_mean": mid,
            "paper_mid_mean": p_mid,
            "late_mean": late,
            "paper_late_mean": p_late,
            "overall_mean": overall_mean,
            "paper_overall_mean": p_overall,
            "drop_val": drop_val,
            "paper_drop_val": p_drop,
            "late_peak_ratio": late_peak_ratio,
            "paper_late_peak_ratio": p_ratio,
            "n_episodes": len(alpha_beta_trajectories),
            "n_steps": max_len,
            "pearson_mean_trace": r_shape,
            **metrics_extra,
        },
        paper_ref={"path": str(curves_path("fig7"))},
        notes=[
            "Fig 7 uses per-step mean across eval episodes vs paper average curve.",
            "mean_below_theta is informational — paper digitized mean can sit above θ.",
        ],
    )
