"""Digitization-anchored gate helpers for Nguyen (paper 2) panels.

Loads normalized ``curves_fig*.json`` under
``artifacts/figures/papers/nguyen/paper_digitization/`` and compares
replication traces with **x-axis windows** (episode index, RL step).

Fig 3 is a scatter/boxplot — no refined time-series digitization yet; gates use
documented paper readouts (~215 / ~295 means) until ``curves_fig3`` exists.
"""
from __future__ import annotations

import functools
import json
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


# ---------------------------------------------------------------------------
# Fig 4 — DSQN training reward (panel a) and episode length (panel b)
# ---------------------------------------------------------------------------
#
# Anchors come from the digitized paper ``Smoothed`` series interpolated onto a
# whole-episode grid (0..499). Our series is smoothed with a centred 20-episode
# moving average on the same grid. Reward gates are scale-free (progress,
# timing milestones, Pearson r, relative gain) because Eq. (7) leaves δ, τ and
# the normalization of d unstated; the one absolute-scale check
# (``reward_scale_raw_d``) is tier 2 / logged only.
#
# Paper reference values quoted in comments below are what
# ``fig4_paper_reference()`` computes from the digitization (oct 3 2026).

FIG4_EPISODES = 500  # paper Fig 4 x-axis: 500 training episodes
FIG4_SMOOTH = 20  # centred moving-average window (episodes) for our curves
FIG4_START_WINDOW = (0, 50)  # "start" level window; paper len 24.91, reward −487.5k
FIG4_PLATEAU_START = 150  # plateau / late window start; paper progress ≥0.96 (reward) from here
FIG4_LATE_LEVEL_LO = 350  # second late window; paper len 8.19 over ep 350–500
FIG4_PEARSON_EARLY_HI = 200  # transition-focused Pearson window (ep 0–200)

# Length starts at the 25-step horizon: paper smoothed ep 0–50 = 24.9 of 25.
FIG4_HORIZON_SLACK = 2.0
# 50%-progress milestone window. Paper t50: length ep 76, reward ep 71. Early side:
# paper t50 − 15 (a 15-ep shift of the paper curve keeps Pearson r ≥ 0.94).
# Late side: the paper text puts the exploration→exploitation transition at ~ep 100.
FIG4_T50_EARLY_SLACK = 15
FIG4_T50_LATEST = 100
# 90%-progress milestone: paper t90 length ep 89, reward ep 86; allow paper + 25.
FIG4_T90_LATE_SLACK = 25
# Pearson r vs the paper smoothed curve. The paper against itself shifted 20 ep
# gives r(0–200) = 0.90 (length) / 0.93 (reward) and r(0–500) = 0.935 / 0.947.
FIG4_PEARSON_0_200_MIN = 0.90
FIG4_PEARSON_0_500_MIN = 0.93
# Late length level and late/early length ratio: ±25 % of the paper value
# (paper len ep 150–500 = 8.57, ep 350–500 = 8.19, ratio 0.344; raw band 7.1–11.5).
FIG4_LEVEL_REL_TOL = 0.25
# Reward relative gain (R̄150–500 − R̄0–50) / |R̄0–50|: paper 0.974.
FIG4_REWARD_GAIN_MIN = 0.85
# Lowest smoothed progress over ep 150–500: paper 0.96 (reward) / 0.92 (length).
FIG4_PLATEAU_HOLD_MIN = 0.80
# Timeouts (length = horizon) after ep 150: paper 0 — raw upper edge peaks at 23.3.
FIG4_LATE_TIMEOUT_MAX = 0.05
# Late smoothed-length slope ep 350–490: paper −0.007 / episode.
FIG4_LATE_SLOPE_WINDOW = (350, 490)
FIG4_LATE_LENGTH_SLOPE_MAX = 0.02
# Tier 2: first-50 reward vs paper −487.5k, meaningful only if d = (αβ − θ)² unnormalized.
FIG4_SCALE_RAW_D_REL_TOL = 0.35

# Early abort (doomed-run check only): progress over ep 120–150 toward the paper's
# late level, scale-free. Paper: length 0.93, reward 0.95 in that window.
FIG4_ABORT_WINDOW = (120, 150)
FIG4_ABORT_EPISODE = FIG4_ABORT_WINDOW[1]
FIG4_ABORT_PROGRESS_MIN = 0.5

# Gate keys, ordered by importance. ``shape`` = transition timing / ep 0–200 shape
# (progress milestones still normalize by the run's own ep 150–500 level);
# ``full`` (ship ``pass``) = all required; ``tier2`` = logged only.
FIG4_LENGTH_SHAPE_KEYS: tuple[str, ...] = (
    "length_t50_timing",
    "length_pearson_0_200",
    "length_start_at_horizon",
)
FIG4_LENGTH_REQUIRED_KEYS: tuple[str, ...] = (
    "length_late_level",
    *FIG4_LENGTH_SHAPE_KEYS,
    "length_pearson_0_500",
    "length_ratio",
    "length_late_plateau_hold",
    "late_timeouts",
    "late_length_no_regression",
)
FIG4_REWARD_SHAPE_KEYS: tuple[str, ...] = (
    "reward_t50_timing",
    "reward_pearson_0_200",
)
FIG4_REWARD_REQUIRED_KEYS: tuple[str, ...] = (
    *FIG4_REWARD_SHAPE_KEYS,
    "reward_pearson_0_500",
    "reward_relative_gain",
    "reward_late_plateau_hold",
)
FIG4_REWARD_TIER2_KEYS: tuple[str, ...] = ("reward_leads_length", "reward_scale_raw_d")
FIG4_LENGTH_TIER2_KEYS: tuple[str, ...] = ()

FIG4_SHAPE_KEYS: dict[str, tuple[str, ...]] = {
    "reward": FIG4_REWARD_SHAPE_KEYS,
    "length": FIG4_LENGTH_SHAPE_KEYS,
}
FIG4_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "reward": FIG4_REWARD_REQUIRED_KEYS,
    "length": FIG4_LENGTH_REQUIRED_KEYS,
}
FIG4_TIER2_KEYS: dict[str, tuple[str, ...]] = {
    "reward": FIG4_REWARD_TIER2_KEYS,
    "length": FIG4_LENGTH_TIER2_KEYS,
}


class Fig4GateKeyError(KeyError):
    """A required Fig 4 gate key was not computed (never treat a missing key as a fail)."""


def require_gate_keys(gates: dict[str, Any], keys: tuple[str, ...], *, group: str) -> None:
    """Raise if any ``keys`` is absent from ``gates`` or not a bool."""
    missing = [k for k in keys if not isinstance(gates.get(k), bool)]
    if missing:
        msg = f"Fig 4 {group} gates not computed: {missing}"
        raise Fig4GateKeyError(msg)


def gate_group_pass(gates: dict[str, Any], keys: tuple[str, ...], *, group: str) -> bool:
    """All ``keys`` true; raises :class:`Fig4GateKeyError` when a key is missing."""
    require_gate_keys(gates, keys, group=group)
    return all(bool(gates[k]) for k in keys)


def centered_smooth(y: list[float] | np.ndarray, window: int = FIG4_SMOOTH) -> np.ndarray:
    """Centred moving average (samples ``i − w//2 … i − w//2 + w − 1``), shrinking at the edges."""
    arr = np.asarray(y, dtype=float)
    n = int(arr.size)
    if n == 0 or window <= 1:
        return arr.copy()
    half = int(window) // 2
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx - half + int(window), 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def raw_outline_edges(stem: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split a digitized ``Raw`` outline into its ``lower`` and ``upper`` edges.

    Fig 4 ``Raw`` was traced as a closed outline of the noisy band: one pass out
    to the right edge, then back. Sorting the points together (``load_refined``)
    interleaves both edges, so split at the rightmost point and sort each pass.
    """
    payload = json.loads(curves_path(stem).read_text(encoding="utf-8"))
    xy = payload["series"]["Raw"]["xy"]
    x = np.asarray(xy["x"], dtype=float)
    y = np.asarray(xy["y"], dtype=float)
    k = int(np.argmax(x))
    passes = []
    for px, py in ((x[: k + 1], y[: k + 1]), (x[k:], y[k:])):
        order = np.argsort(px, kind="mergesort")
        passes.append((px[order], py[order]))
    lower, upper = sorted(passes, key=lambda p: float(np.mean(p[1])))
    return {"lower": lower, "upper": upper}


def _on_grid(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    return np.interp(np.arange(n, dtype=float), x[order], y[order])


def _on_grid_traced(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    grid = _on_grid(x, y, n)
    episodes = np.arange(n, dtype=float)
    grid[(episodes < float(np.min(x))) | (episodes > float(np.max(x)))] = np.nan
    return grid


def _nanmean(seg: np.ndarray) -> float:
    return float(np.nanmean(seg)) if np.any(np.isfinite(seg)) else float("nan")


def _wmean(arr: np.ndarray, lo: int, hi: int | None = None) -> float:
    seg = arr[lo : (arr.size if hi is None else min(hi, arr.size))]
    return float(np.mean(seg)) if seg.size else float("nan")


def _progress(smooth: np.ndarray, start: float, end: float) -> np.ndarray:
    denom = end - start
    if not (np.isfinite(start) and np.isfinite(end)) or abs(denom) <= 1e-9 * max(abs(start), 1.0):
        return np.full(smooth.shape, np.nan)
    return (smooth - start) / denom


def _first_reach(progress: np.ndarray, level: float) -> int | None:
    idx = np.flatnonzero(progress >= level)
    return int(idx[0]) if idx.size else None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or a.size != b.size or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _slope(y: np.ndarray, lo: int, hi: int) -> float:
    seg = y[lo : min(hi, y.size)]
    if seg.size < 2:
        return float("nan")
    return float(np.polyfit(np.arange(lo, lo + seg.size, dtype=float), seg, 1)[0])


def _curve_stats(raw: np.ndarray, smooth: np.ndarray) -> dict[str, Any]:
    """Shared stats for our series and the paper grid (``raw`` = per-episode values)."""
    s0, s1 = FIG4_START_WINDOW
    start = _wmean(smooth, s0, s1)
    end = _wmean(smooth, FIG4_PLATEAU_START)
    prog = _progress(smooth, start, end)
    late = prog[FIG4_PLATEAU_START:]
    raw_start = _wmean(raw, s0, s1)
    raw_late = _wmean(raw, FIG4_PLATEAU_START)
    return {
        "progress": prog,
        "start_smoothed": start,
        "end_smoothed": end,
        "t10": _first_reach(prog, 0.10),
        "t50": _first_reach(prog, 0.50),
        "t90": _first_reach(prog, 0.90),
        "plateau_min_progress": float(np.nanmin(late)) if late.size and np.any(np.isfinite(late)) else float("nan"),
        "mean_0_50": raw_start,
        "mean_150_end": raw_late,
        "mean_350_end": _wmean(raw, FIG4_LATE_LEVEL_LO),
        "late_ratio": raw_late / raw_start if abs(raw_start) > 1e-12 else float("nan"),
        "relative_gain": (raw_late - raw_start) / abs(raw_start) if abs(raw_start) > 1e-12 else float("nan"),
    }


def _abort_progress(raw: np.ndarray, *, target_ratio: float) -> float:
    """Progress over the abort window toward ``start × target_ratio`` (scale-free)."""
    s0, s1 = FIG4_START_WINDOW
    lo, hi = FIG4_ABORT_WINDOW
    start = _wmean(raw, s0, s1)
    window = _wmean(raw, lo, hi)
    target = start * target_ratio
    denom = target - start
    if not (np.isfinite(start) and np.isfinite(window)) or abs(denom) <= 1e-9 * max(abs(start), 1.0):
        return float("nan")
    return float((window - start) / denom)


@functools.lru_cache(maxsize=1)
def _fig4_paper_cached() -> dict[str, Any]:
    n = FIG4_EPISODES
    rx, ry = _pick_series(load_curves("fig4_reward"), "Smoothed")
    lx, ly = _pick_series(load_curves("fig4_length"), "Smoothed")
    rs = _on_grid(rx, ry, n)
    ls = _on_grid(lx, ly, n)
    ref: dict[str, Any] = {"reward_grid": rs, "length_grid": ls}
    ref["reward"] = _curve_stats(rs, rs)
    ref["length"] = _curve_stats(ls, ls)
    ref["length"]["slope_350_490"] = _slope(ls, *FIG4_LATE_SLOPE_WINDOW)
    ref["reward"]["abort_progress"] = _abort_progress(rs, target_ratio=ref["reward"]["late_ratio"])
    ref["length"]["abort_progress"] = _abort_progress(ls, target_ratio=ref["length"]["late_ratio"])
    band: dict[str, Any] = {}
    for name, stem in (("reward", "fig4_reward"), ("length", "fig4_length")):
        edges = raw_outline_edges(stem)
        # NaN outside each edge's traced x-range (no flat extrapolation).
        lo_g = _on_grid_traced(*edges["lower"], n)
        hi_g = _on_grid_traced(*edges["upper"], n)
        band[name] = {
            f"{a}_{b}": [_nanmean(lo_g[a:b]), _nanmean(hi_g[a:b])]
            for a, b in ((0, 50), (50, 100), (80, 100), (100, 150), (150, 500), (350, 500))
        }
        band[name]["upper_max_150_500"] = float(np.nanmax(hi_g[FIG4_PLATEAU_START:]))
    ref["raw_band"] = band
    return ref


def fig4_paper_reference() -> dict[str, Any]:
    """Digitized paper Fig 4 stats on a whole-episode grid (cached; arrays included)."""
    return _fig4_paper_cached()


def _jsonable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}


def fig4_paper_summary() -> dict[str, Any]:
    """JSON-safe paper anchors for manifests."""
    ref = fig4_paper_reference()
    return {
        "reward_curves": str(curves_path("fig4_reward")),
        "length_curves": str(curves_path("fig4_length")),
        "grid_episodes": FIG4_EPISODES,
        "reward": _jsonable_stats(ref["reward"]),
        "length": _jsonable_stats(ref["length"]),
        "raw_band": ref["raw_band"],
    }


def _t_milestone_ok(ours: dict[str, Any], paper: dict[str, Any], *, direction_ok: bool) -> bool:
    t50, t90 = ours["t50"], ours["t90"]
    p50, p90 = paper["t50"], paper["t90"]
    if not direction_ok or t50 is None or t90 is None or p50 is None or p90 is None:
        return False
    return bool(p50 - FIG4_T50_EARLY_SLACK <= t50 <= FIG4_T50_LATEST and t90 <= p90 + FIG4_T90_LATE_SLACK)


def _finite_ge(value: float, floor: float) -> bool:
    return bool(np.isfinite(value) and value >= floor)


def fig4_gates(
    episode_rewards: list[float] | np.ndarray,
    episode_lengths: list[int] | np.ndarray,
    *,
    max_episode_steps: int = 25,
    smooth_window: int = FIG4_SMOOTH,
) -> dict[str, Any]:
    """Fig 4 reward + length gates vs the digitized paper curves.

    Returns ``{"reward": {...}, "length": {...}, "shape_pass", "pass", ...}``.
    Each group holds its required gate bools at top level, ``tier2`` (logged only)
    and ``metrics``. Every required key is always computed — short series fail
    with ``False`` rather than dropping keys.
    """
    rewards = np.asarray(episode_rewards, dtype=float)
    lengths = np.asarray(episode_lengths, dtype=float)
    n = int(min(rewards.size, lengths.size))
    rewards = rewards[:n]
    lengths = lengths[:n]
    ref = fig4_paper_reference()
    p_r, p_l = ref["reward"], ref["length"]

    rs = centered_smooth(rewards, smooth_window)
    ls = centered_smooth(lengths, smooth_window)
    our_r = _curve_stats(rewards, rs)
    our_l = _curve_stats(lengths, ls)

    def pearson(ours_s: np.ndarray, paper_s: np.ndarray, hi: int) -> float:
        if n < hi:
            return float("nan")
        return _pearson(ours_s[:hi], paper_s[:hi])

    r_len_200 = pearson(ls, ref["length_grid"], FIG4_PEARSON_EARLY_HI)
    r_len_500 = pearson(ls, ref["length_grid"], FIG4_EPISODES)
    r_rew_200 = pearson(rs, ref["reward_grid"], FIG4_PEARSON_EARLY_HI)
    r_rew_500 = pearson(rs, ref["reward_grid"], FIG4_EPISODES)

    late_raw = lengths[FIG4_PLATEAU_START:]
    timeout_rate = (
        float(np.mean(late_raw >= float(max_episode_steps) - 0.5)) if late_raw.size else float("nan")
    )
    len_slope = _slope(ls, *FIG4_LATE_SLOPE_WINDOW)
    start_median = float(np.median(lengths[: FIG4_START_WINDOW[1]])) if n else float("nan")

    length_gates = {
        "length_late_level": bool(
            rel_close(our_l["mean_150_end"], p_l["mean_150_end"], tol=FIG4_LEVEL_REL_TOL)
            and rel_close(our_l["mean_350_end"], p_l["mean_350_end"], tol=FIG4_LEVEL_REL_TOL)
        ),
        "length_t50_timing": _t_milestone_ok(
            our_l, p_l, direction_ok=bool(our_l["end_smoothed"] < our_l["start_smoothed"])
        ),
        "length_pearson_0_200": _finite_ge(r_len_200, FIG4_PEARSON_0_200_MIN),
        "length_start_at_horizon": _finite_ge(start_median, float(max_episode_steps) - FIG4_HORIZON_SLACK),
        "length_pearson_0_500": _finite_ge(r_len_500, FIG4_PEARSON_0_500_MIN),
        "length_ratio": rel_close(our_l["late_ratio"], p_l["late_ratio"], tol=FIG4_LEVEL_REL_TOL),
        "length_late_plateau_hold": _finite_ge(our_l["plateau_min_progress"], FIG4_PLATEAU_HOLD_MIN),
        "late_timeouts": bool(np.isfinite(timeout_rate) and timeout_rate <= FIG4_LATE_TIMEOUT_MAX),
        "late_length_no_regression": bool(np.isfinite(len_slope) and len_slope <= FIG4_LATE_LENGTH_SLOPE_MAX),
    }
    reward_gates = {
        "reward_t50_timing": _t_milestone_ok(
            our_r, p_r, direction_ok=bool(our_r["end_smoothed"] > our_r["start_smoothed"])
        ),
        "reward_pearson_0_200": _finite_ge(r_rew_200, FIG4_PEARSON_0_200_MIN),
        "reward_pearson_0_500": _finite_ge(r_rew_500, FIG4_PEARSON_0_500_MIN),
        "reward_relative_gain": bool(
            np.isfinite(our_r["mean_0_50"])
            and our_r["mean_0_50"] < 0.0
            and _finite_ge(our_r["relative_gain"], FIG4_REWARD_GAIN_MIN)
        ),
        "reward_late_plateau_hold": _finite_ge(our_r["plateau_min_progress"], FIG4_PLATEAU_HOLD_MIN),
    }
    reward_tier2 = {
        "reward_leads_length": bool(
            our_r["t10"] is not None and our_l["t10"] is not None and our_r["t10"] < our_l["t10"]
        ),
        "reward_scale_raw_d": rel_close(our_r["mean_0_50"], p_r["mean_0_50"], tol=FIG4_SCALE_RAW_D_REL_TOL),
    }
    length_tier2: dict[str, bool] = {}

    reward_metrics = {
        **_jsonable_stats(our_r),
        "pearson_0_200": r_rew_200,
        "pearson_0_500": r_rew_500,
    }
    length_metrics = {
        **_jsonable_stats(our_l),
        "pearson_0_200": r_len_200,
        "pearson_0_500": r_len_500,
        "start_median_0_50": start_median,
        "timeout_rate_150_end": timeout_rate,
        "slope_350_490": len_slope,
    }

    out: dict[str, Any] = {"n_episodes": n}
    for name, gates, tier2, metrics in (
        ("reward", reward_gates, reward_tier2, reward_metrics),
        ("length", length_gates, length_tier2, length_metrics),
    ):
        require_gate_keys(gates, FIG4_REQUIRED_KEYS[name], group=name)
        require_gate_keys(tier2, FIG4_TIER2_KEYS[name], group=f"{name} tier2")
        group = dict(gates)
        group["shape_pass"] = gate_group_pass(gates, FIG4_SHAPE_KEYS[name], group=name)
        group["pass"] = gate_group_pass(gates, FIG4_REQUIRED_KEYS[name], group=name)
        group["failed"] = [k for k in FIG4_REQUIRED_KEYS[name] if not gates[k]]
        group["tier2"] = tier2
        group["metrics"] = metrics
        out[name] = group
    out["shape_pass"] = bool(out["reward"]["shape_pass"] and out["length"]["shape_pass"])
    out["pass"] = bool(out["reward"]["pass"] and out["length"]["pass"])
    out["required_keys"] = {k: list(v) for k, v in FIG4_REQUIRED_KEYS.items()}
    out["shape_keys"] = {k: list(v) for k, v in FIG4_SHAPE_KEYS.items()}
    out["tier2_keys"] = {k: list(v) for k, v in FIG4_TIER2_KEYS.items()}
    out["smooth_window"] = int(smooth_window)
    out["paper_ref"] = fig4_paper_summary()
    return out


def fig4_abort_check(
    episode_rewards: list[float] | np.ndarray,
    episode_lengths: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Doomed-run check for the early abort (only this — never the full gate set).

    Once ep 120–150 exist: abort when progress from the ep 0–50 level toward the
    paper's late level (length ratio 0.344, reward gain 0.974 — scale-free) is
    below ``FIG4_ABORT_PROGRESS_MIN`` for length or reward. The paper is at ~0.93
    (length) / ~0.95 (reward) in that window. A non-finite reward progress (e.g.
    non-negative early reward) is reported but does not abort.
    """
    rewards = np.asarray(episode_rewards, dtype=float)
    lengths = np.asarray(episode_lengths, dtype=float)
    n = int(min(rewards.size, lengths.size))
    lo, hi = FIG4_ABORT_WINDOW
    probe: dict[str, Any] = {
        "completed_episodes": n,
        "abort_window": [lo, hi],
        "decidable": n >= hi,
        "abort": False,
        "failed": [],
    }
    if not probe["decidable"]:
        return probe
    ref = fig4_paper_reference()
    p_len = _abort_progress(lengths, target_ratio=ref["length"]["late_ratio"])
    p_rew = _abort_progress(rewards, target_ratio=ref["reward"]["late_ratio"])
    failed: list[str] = []
    if not _finite_ge(p_len, FIG4_ABORT_PROGRESS_MIN):
        failed.append("abort_length_progress_120_150")
    if np.isfinite(p_rew) and p_rew < FIG4_ABORT_PROGRESS_MIN:
        failed.append("abort_reward_progress_120_150")
    probe.update(
        {
            "length_progress_120_150": p_len,
            "reward_progress_120_150": p_rew,
            "paper_length_progress_120_150": ref["length"]["abort_progress"],
            "paper_reward_progress_120_150": ref["reward"]["abort_progress"],
            "progress_min": FIG4_ABORT_PROGRESS_MIN,
            "failed": failed,
            "abort": bool(failed),
        }
    )
    return probe


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
