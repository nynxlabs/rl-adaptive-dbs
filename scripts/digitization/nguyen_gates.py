"""Digitization-anchored gate helpers for Nguyen (paper 2) panels.

Loads normalized ``curves_fig*.json`` under
``artifacts/figures/papers/nguyen/paper_digitization/`` and compares
replication traces with **x-axis windows** (episode index, RL step).

Fig 3 is a scatter/boxplot: its gates read the box statistics in
``curves_fig3_boxplot.json`` (``scripts/digitization/nguyen_fig3_boxplot.py``).
"""
from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

from paper_gates import (
    DEFAULT_REL_TOL,
    _gate_pack,
    load_refined,
    pearson_on_ref_x,
    rel_close,
    window_mean,
)

ARTIFACT_ROOT = Path("artifacts/figures/papers/nguyen/paper_digitization")

# Paper Fig 3 means as read off panel (a) (curves_fig3_boxplot.json: 216.0 / 296.7).
PAPER_FIG3_PD_OFF_MEAN = 216.0
PAPER_FIG3_PD_ON_MEAN = 296.7
THETA = 150.0

INIT_AMP = 300.0
INIT_FREQ = 40.0
INIT_PW = 0.3


def curves_path(stem: str) -> Path:
    """Canonical normalized curves JSON for a Nguyen sub-panel."""
    return ARTIFACT_ROOT / f"curves_{stem}.json"


# Raw per-episode traces whose one-episode spikes must survive loading (load_refined).
UNBINNED_STEMS = frozenset({"fig5_spikes", "fig5_energy"})


def load_curves(stem: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from ``curves_{stem}.json``."""
    return load_refined(curves_path(stem), bin_near_duplicates=stem not in UNBINNED_STEMS)


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
    for key, value in dig_report.get("report", {}).items():
        out[f"{prefix}{key}"] = bool(value)
    out["paper_gate_metrics"] = dig_report.get("metrics", {})
    out["paper_ref"] = dig_report.get("paper_ref", {})
    out["paper_notes"] = list(dig_report.get("notes", []))
    h_pass = bool(heuristic.get("pass", False))
    d_pass = bool(dig_report.get("pass", True))
    out["pass"] = h_pass and d_pass
    return out


FIG3_STATS = ("q1", "median", "q3", "mean")
FIG3_REL_TOL = 0.10


@functools.lru_cache(maxsize=1)
def fig3_paper_boxplot() -> dict[str, dict[str, float]]:
    """Paper Fig 3 box statistics + means (``scripts/digitization/nguyen_fig3_boxplot.py``)."""
    data = json.loads(curves_path("fig3_boxplot").read_text(encoding="utf-8"))
    return data["groups"]


def fig3_gates(
    samples: dict[str, Any],
    *,
    rel_tol: float = FIG3_REL_TOL,
) -> dict[str, Any]:
    """Fig 3 distributions vs the digitized paper boxplot (panel b) and mean lines (panel a).

    Required: PD On above PD Off; Q1 / median / Q3 / mean of each group within ±10 % of the
    paper (pixel read ~1.4 units; 500-sample quartiles move ~2 %); and PD Off Q1 just above
    θ = 150, the paper's stated reason for θ ("the first quartile … is marginally higher
    than 150" — the PD On Q1 is ~264, so the sentence can only mean PD Off).
    """
    paper = fig3_paper_boxplot()
    ours: dict[str, dict[str, float]] = {}
    for group in ("pd_off", "pd_on"):
        arr = np.asarray(samples[group], dtype=float)
        ours[group] = {
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "q3": float(np.percentile(arr, 75)),
            "mean": float(np.mean(arr)),
        }
    gates: dict[str, bool] = {
        "ordering_pd_on_above_pd_off": ours["pd_on"]["median"] > ours["pd_off"]["median"],
    }
    metrics: dict[str, Any] = {}
    for group in ("pd_off", "pd_on"):
        for stat in FIG3_STATS:
            key = f"{group}_{stat}"
            gates[f"{key}_near_paper"] = rel_close(ours[group][stat], paper[group][stat], tol=rel_tol)
            metrics[key] = ours[group][stat]
            metrics[f"paper_{key}"] = paper[group][stat]
    gates["pd_off_q1_marginally_above_theta"] = THETA < ours["pd_off"]["q1"] <= 1.25 * THETA
    return _gate_pack(
        gates,
        metrics,
        paper_ref={
            "path": str(curves_path("fig3_boxplot")),
            "note": "pixel read of the paper boxplot and mean lines (nguyen_fig3_boxplot.py)",
        },
        notes=[f"level gates ±{rel_tol:.0%} of the paper; θ gate: 150 < PD Off Q1 ≤ 187.5"],
    )


@dataclasses.dataclass(frozen=True)
class Check:
    """One calibrated comparison of our metric with the paper's.

    ``metric`` names the value in both metric dicts (``group.name`` for Fig 4).
    ``kind``: ``abs`` → ``|ours − paper| ≤ tol``; ``ge`` → ``ours ≥ paper − tol``;
    ``le`` → ``ours ≤ paper + tol``. A gate key may combine several checks (all
    must hold). ``note`` is the unit / paper value for the calibration table.
    """

    gate: str
    metric: str
    kind: str
    tol: float
    note: str = ""


def check_deviation(check: Check, ours: float | None, paper: float | None) -> float:
    """Signed shortfall in ``tol`` units of the metric: the check passes iff this ≤ ``tol``."""
    if ours is None or paper is None or not (np.isfinite(ours) and np.isfinite(paper)):
        return float("inf")
    d = float(ours) - float(paper)
    if check.kind == "abs":
        return abs(d)
    if check.kind == "ge":
        return -d
    if check.kind == "le":
        return d
    msg = f"unknown check kind {check.kind!r}"
    raise ValueError(msg)


def apply_checks(
    checks: tuple[Check, ...],
    ours: dict[str, Any],
    paper: dict[str, Any],
) -> dict[str, bool]:
    """Gate bools from ``checks`` (a key passes only if all of its checks pass)."""
    gates: dict[str, bool] = {}
    for c in checks:
        ok = check_deviation(c, ours.get(c.metric), paper.get(c.metric)) <= c.tol
        gates[c.gate] = bool(gates.get(c.gate, True) and ok)
    return gates


def _flat_metrics(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {f"{g}.{k}": v for g, d in groups.items() for k, v in d.items()}


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

# Calibrated tolerances (scripts/probes/nguyen_gate_calibration.py). The noise model:
# a "faithful run" is the digitized paper curve on the episode grid plus one seed's
# deviation from its config's mean curve (inflated by √(N/(N−1))). Each tolerance is
# the smallest clean number (2 significant figures; milestones in steps of 5 episodes)
# at which ≥ 90 % of those synthetic runs pass the gate key (a key with two checks
# widens both together, by the smallest common factor). Pools (oct 17 2026): wu1500 ×13,
# s5d1 ×13, audit1-base / -net1 / -net1seq / -net1seq100 ×8 each — 58 runs (45 with
# α–β finals for Fig 6).
# Pearson checks compare with the paper (r = 1 against itself), so tol = 1 − r_min.
FIG4_CHECKS: tuple[Check, ...] = (
    # Length (panel b): absolute residuals (steps).
    Check("length_late_level", "length.mean_150_end", "abs", 1.7, "steps; paper 8.57"),
    Check("length_late_level", "length.mean_350_end", "abs", 1.9, "steps; paper 8.19"),
    Check("length_t50_timing", "length.t50", "abs", 10.0, "episodes; paper 76"),
    Check("length_t50_timing", "length.t90", "abs", 60.0, "episodes; paper 89"),
    Check("length_pearson_0_200", "length.pearson_0_200", "ge", 0.059, "r ≥ 0.941"),
    Check("length_start_at_horizon", "length.start_median_0_50", "ge", 0.5, "steps; paper 24.9 (integer lengths: half a step)"),
    Check("length_pearson_0_500", "length.pearson_0_500", "ge", 0.075, "r ≥ 0.925"),
    Check("length_ratio", "length.late_ratio", "abs", 0.087, "paper 0.344"),
    Check("length_late_plateau_hold", "length.plateau_min_progress", "ge", 0.3, "paper 0.92"),
    Check("late_timeouts", "length.timeout_rate_150_end", "le", 0.0086, "paper 0 (≤ 3 of 350 episodes)"),
    Check("length_late_trend", "length.late_trend", "abs", 0.15, "L̄350–500 / L̄150–350; paper 0.925"),
    # Reward (panel a): absolute residuals (our reward scale matches the paper's).
    Check("reward_t50_timing", "reward.t50", "abs", 45.0, "episodes; paper 71"),
    Check("reward_t50_timing", "reward.t90", "abs", 35.0, "episodes; paper 86"),
    Check("reward_pearson_0_200", "reward.pearson_0_200", "ge", 0.062, "r ≥ 0.938"),
    Check("reward_pearson_0_500", "reward.pearson_0_500", "ge", 0.058, "r ≥ 0.942"),
    Check("reward_relative_gain", "reward.relative_gain", "abs", 0.026, "paper 0.974"),
    Check("reward_late_plateau_hold", "reward.plateau_min_progress", "ge", 0.058, "paper 0.96"),
    Check("reward_leads_length", "reward.t10_lead", "ge", 35.0, "length t10 − reward t10; paper 36"),
)
# Tier 2: first-50 reward vs paper −487.5k, meaningful only if d = (αβ − θ)² unnormalized.
FIG4_SCALE_RAW_D_REL_TOL = 0.35
# Late slope window, logged in the metrics only (paper −0.007 / episode).
FIG4_LATE_SLOPE_WINDOW = (350, 490)
FIG4_LATE_TREND_SPLIT = 350  # late_trend = L̄350–500 / L̄150–350

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
    "length_late_trend",
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
    "reward_leads_length",
)
FIG4_REWARD_TIER2_KEYS: tuple[str, ...] = ("reward_scale_raw_d",)
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


def _finite_ge(value: float, floor: float) -> bool:
    return bool(np.isfinite(value) and value >= floor)


def _late_trend(raw: np.ndarray) -> float:
    """``mean(ep 350–500) / mean(ep 150–350)`` — late drift direction (paper length 0.925)."""
    a = _wmean(raw, FIG4_PLATEAU_START, FIG4_LATE_TREND_SPLIT)
    b = _wmean(raw, FIG4_LATE_TREND_SPLIT, FIG4_EPISODES)
    return b / a if np.isfinite(a) and abs(a) > 1e-12 else float("nan")


@functools.lru_cache(maxsize=1)
def _fig4_paper_metrics_cached() -> tuple[tuple[str, Any], ...]:
    ref = fig4_paper_reference()
    p_r, p_l = ref["reward"], ref["length"]
    ls = ref["length_grid"]
    length = {
        **_jsonable_stats(p_l),
        "pearson_0_200": 1.0,
        "pearson_0_500": 1.0,
        "start_median_0_50": float(np.median(ls[: FIG4_START_WINDOW[1]])),
        "timeout_rate_150_end": 0.0,
        "late_trend": _late_trend(ls),
    }
    reward = {
        **_jsonable_stats(p_r),
        "pearson_0_200": 1.0,
        "pearson_0_500": 1.0,
        "t10_lead": float(p_l["t10"] - p_r["t10"]),
    }
    return tuple(_flat_metrics({"reward": reward, "length": length}).items())


def fig4_paper_metrics(*, max_episode_steps: int = 25) -> dict[str, Any]:
    """Paper values of every ``FIG4_CHECKS`` metric (``group.name`` keys)."""
    out = dict(_fig4_paper_metrics_cached())
    # Paper smoothed start reads 24.9–25.9 (trace overshoot); a run cannot exceed the horizon.
    out["length.start_median_0_50"] = min(out["length.start_median_0_50"], float(max_episode_steps))
    return out


def fig4_gates(
    episode_rewards: list[float] | np.ndarray,
    episode_lengths: list[int] | np.ndarray,
    *,
    max_episode_steps: int = 25,
    smooth_window: int = FIG4_SMOOTH,
) -> dict[str, Any]:
    """Fig 4 reward + length gates vs the digitized paper curves (``FIG4_CHECKS``).

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
    p_r = ref["reward"]

    rs = centered_smooth(rewards, smooth_window)
    ls = centered_smooth(lengths, smooth_window)
    our_r = _curve_stats(rewards, rs)
    our_l = _curve_stats(lengths, ls)
    # Milestones only count when the curve moves the paper's way.
    if not our_l["end_smoothed"] < our_l["start_smoothed"]:
        our_l.update({"t10": None, "t50": None, "t90": None})
    if not our_r["end_smoothed"] > our_r["start_smoothed"]:
        our_r.update({"t10": None, "t50": None, "t90": None})

    def pearson(ours_s: np.ndarray, paper_s: np.ndarray, hi: int) -> float:
        if n < hi:
            return float("nan")
        return _pearson(ours_s[:hi], paper_s[:hi])

    late_raw = lengths[FIG4_PLATEAU_START:]
    timeout_rate = (
        float(np.mean(late_raw >= float(max_episode_steps) - 0.5)) if late_raw.size else float("nan")
    )
    full = n >= FIG4_EPISODES
    length_metrics = {
        **_jsonable_stats(our_l),
        "pearson_0_200": pearson(ls, ref["length_grid"], FIG4_PEARSON_EARLY_HI),
        "pearson_0_500": pearson(ls, ref["length_grid"], FIG4_EPISODES),
        "start_median_0_50": float(np.median(lengths[: FIG4_START_WINDOW[1]])) if n else float("nan"),
        "timeout_rate_150_end": timeout_rate if full else float("nan"),
        "slope_350_490": _slope(ls, *FIG4_LATE_SLOPE_WINDOW),
        "late_trend": _late_trend(lengths) if full else float("nan"),
    }
    reward_metrics = {
        **_jsonable_stats(our_r),
        "pearson_0_200": pearson(rs, ref["reward_grid"], FIG4_PEARSON_EARLY_HI),
        "pearson_0_500": pearson(rs, ref["reward_grid"], FIG4_EPISODES),
        "t10_lead": (
            float(our_l["t10"] - our_r["t10"])
            if our_l["t10"] is not None and our_r["t10"] is not None
            else float("nan")
        ),
    }
    if not full:
        # Late-window levels need the whole 500-episode run.
        for m in (length_metrics, reward_metrics):
            for key in ("mean_150_end", "mean_350_end", "late_ratio", "relative_gain", "plateau_min_progress"):
                m[key] = float("nan")
    ours = _flat_metrics({"reward": reward_metrics, "length": length_metrics})
    paper = fig4_paper_metrics(max_episode_steps=max_episode_steps)
    checked = apply_checks(FIG4_CHECKS, ours, paper)
    # Eq. (7) rewards are negative costs: the start level must be below zero.
    checked["reward_relative_gain"] = bool(
        checked["reward_relative_gain"] and np.isfinite(our_r["mean_0_50"]) and our_r["mean_0_50"] < 0.0
    )
    length_gates = {k: checked[k] for k in FIG4_LENGTH_REQUIRED_KEYS}
    reward_gates = {k: checked[k] for k in FIG4_REWARD_REQUIRED_KEYS}
    reward_tier2 = {
        "reward_scale_raw_d": rel_close(our_r["mean_0_50"], p_r["mean_0_50"], tol=FIG4_SCALE_RAW_D_REL_TOL),
    }
    length_tier2: dict[str, bool] = {}

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
    out["checks"] = {c.metric: {"gate": c.gate, "kind": c.kind, "tol": c.tol} for c in FIG4_CHECKS}
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


# ---------------------------------------------------------------------------
# Figs 5–7 — calibrated, digitization-anchored checks
# ---------------------------------------------------------------------------
#
# Same noise model and 90 % rule as ``FIG4_CHECKS``
# (scripts/probes/nguyen_gate_calibration.py). Windows are whole-episode
# ``[lo, hi)`` slices of the paper curve interpolated onto the episode grid.

FIG5_CHECKS: tuple[Check, ...] = (
    # Spike count: relative residuals (the paper never defines its count).
    Check("spike_trend_near_paper", "spike_trend", "abs", 0.099, "S̄350–500 / S̄0–50; paper 1.00"),
    # DBS energy: relative residuals (our energy scale differs from the paper's).
    Check("energy_pearson_0_500", "energy_pearson_0_500", "ge", 0.21, "r ≥ 0.79"),
    Check("energy_rise_timing", "energy_t50", "abs", 25.0, "episodes; paper 73"),
    Check("energy_rise_timing", "energy_t90", "abs", 40.0, "episodes; paper 88"),
    Check("energy_mid_ramp_near_paper", "energy_mid_ramp", "abs", 0.23, "Ē55–75 / Ē0–50; paper 1.15"),
    Check("energy_trend_near_paper", "energy_trend", "abs", 0.6, "Ē350–500 / Ē0–50; paper 1.80"),
    Check("energy_overshoot", "energy_overshoot", "abs", 0.087, "Ē150–350 / Ē350–500; paper 1.16"),
)
FIG5_LEVEL_REPORT_KEYS: tuple[str, ...] = (
    "spike_early_near_paper",
    "spike_mid_near_paper",
    "spike_late_near_paper",
    "spike_mean_near_paper",
    "spike_stays_near_800",
    "energy_early_near_paper",
    "energy_mid_near_paper",
    "energy_late_near_paper",
    "energy_mean_near_paper",
)
# Superseded by FIG5_CHECKS; kept as logged rows.
FIG5_SUPERSEDED_REPORT_KEYS: tuple[str, ...] = (
    "energy_monotonic_rise",
    "energy_late_above_early",
    "spike_series_has_variance",
    "energy_not_constant",
)
FIG5_REPORT_KEYS: tuple[str, ...] = (*FIG5_LEVEL_REPORT_KEYS, *FIG5_SUPERSEDED_REPORT_KEYS)
FIG5_REQUIRED_KEYS: tuple[str, ...] = tuple(dict.fromkeys(c.gate for c in FIG5_CHECKS))

FIG6_AB_CHECKS: tuple[Check, ...] = (
    # α–β (episode-final reading): absolute residuals (same units, same θ).
    Check("alpha_beta_early_near_paper", "ab_early", "abs", 95.0, "first traced ep–50; paper 289"),
    Check("alpha_beta_mid_near_paper", "ab_mid", "abs", 52.0, "ep 50–100; paper 221"),
    Check("alpha_beta_post100_near_paper", "ab_post100", "abs", 6.4, "ep 100–250; paper 110"),
    Check("alpha_beta_late_near_paper", "ab_late", "abs", 5.3, "ep 350–500; paper 109"),
    Check("alpha_beta_mean_near_paper", "ab_mean", "abs", 11.0, "first traced ep–500; paper 130"),
    Check("alpha_beta_pearson", "ab_pearson", "ge", 0.091, "r ≥ 0.909"),
    Check("alpha_beta_t50_timing", "ab_t50", "abs", 35.0, "episodes; paper 87"),
    Check("alpha_beta_drop_fraction", "ab_drop_fraction", "abs", 0.34, "(early − mid) / (early − late); paper 0.37"),
    Check("alpha_beta_trend_near_paper", "ab_trend", "abs", 0.17, "late / early; paper 0.38"),
    Check("alpha_beta_drop_magnitude_near_paper", "ab_drop", "abs", 93.0, "early − late; paper 180"),
    Check("alpha_beta_late_cv", "ab_late_cv", "abs", 0.025, "std / mean ep 350–500; paper 0.16"),
)
FIG6_FREQ_CHECKS: tuple[Check, ...] = (
    # Frequency: relative residuals (our levels sit below the paper's).
    Check("freq_early_near_paper", "freq_early", "abs", 42.0, "Hz, ep 0–50; paper 59.1"),
    Check("freq_mid_near_paper", "freq_mid", "abs", 37.0, "Hz, ep 50–100; paper 93.7"),
    Check("freq_late_near_paper", "freq_late", "abs", 6.2, "Hz, ep 350–500; paper 81.3"),
    Check("freq_t50_timing", "freq_t50", "abs", 45.0, "episodes; paper 26"),
    Check("freq_pearson", "freq_pearson", "ge", 0.68, "r ≥ 0.32"),
    Check("freq_late_cv", "freq_late_cv", "abs", 0.025, "std / mean ep 350–500; paper 0.076"),
)
FIG6_CHECKS: tuple[Check, ...] = (*FIG6_AB_CHECKS, *FIG6_FREQ_CHECKS)
FIG6_THETA_KEYS: tuple[str, ...] = (
    "alpha_beta_early_above_theta",
    "alpha_beta_post100_below_theta",
    "alpha_beta_late_below_theta",
)
FIG6_REQUIRED_KEYS: tuple[str, ...] = (
    *FIG6_THETA_KEYS,
    *dict.fromkeys(c.gate for c in FIG6_CHECKS),
)
# Pending maintainer decision: amplitude / pulse width do not move α–β in this plant.
FIG6_PARAM_REPORT_KEYS: tuple[str, ...] = (
    "amp_left_init",
    "pw_left_init",
    "params_left_init",
    "amp_late_near_paper",
    "pw_late_near_paper",
    "late_params_stable",
)
# Old heuristics, superseded by FIG6_CHECKS; kept as logged rows.
FIG6_SUPERSEDED_REPORT_KEYS: tuple[str, ...] = (
    "alpha_beta_drops_by_100",
    "alpha_beta_monotonic_drop",
    "alpha_beta_series_has_variance",
    "alpha_beta_late_stable",
    "alpha_beta_decreases_like_paper",
    "freq_left_init",
    "freq_late_stable",
)
FIG6_REPORT_KEYS: tuple[str, ...] = (*FIG6_PARAM_REPORT_KEYS, *FIG6_SUPERSEDED_REPORT_KEYS)

FIG5_6_EPISODES = 500
FIG6_FREQ_START_WINDOW = (0, 10)  # frequency starts at the 40 Hz init, so a short window
FIG6_LATE_LO = 350


def _with_report(pack: dict[str, Any], report: dict[str, bool]) -> dict[str, Any]:
    """Attach logged-only gate rows; they never change ``pass``."""
    pack["report"] = {k: bool(v) for k, v in report.items()}
    return pack


def _paper_grid(stem: str, *aliases: str, traced: bool = False, n: int = FIG5_6_EPISODES) -> np.ndarray:
    x, y = _pick_series(load_curves(stem), *aliases)
    return _on_grid_traced(x, y, n) if traced else _on_grid(x, y, n)


def _nan_wmean(arr: np.ndarray, lo: int, hi: int) -> float:
    return _nanmean(arr[lo : min(hi, arr.size)]) if arr.size > lo else float("nan")


def _milestone(smooth: np.ndarray, start_w: tuple[int, int], end_w: tuple[int, int], level: float) -> int | None:
    start = _nan_wmean(smooth, *start_w)
    end = _nan_wmean(smooth, *end_w)
    prog = _progress(smooth, start, end)
    prog = np.where(np.isfinite(prog), prog, -np.inf)
    return _first_reach(prog, level)


def _ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12 else float("nan")


def _late_cv(arr: np.ndarray, lo: int = FIG6_LATE_LO) -> float:
    seg = arr[lo:]
    seg = seg[np.isfinite(seg)]
    return float(np.std(seg) / np.mean(seg)) if seg.size > 2 and abs(np.mean(seg)) > 1e-12 else float("nan")


def _pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return _pearson(a[m], b[m])


def _fig5_metrics(spikes: np.ndarray, energies: np.ndarray, *, is_paper: bool = False) -> dict[str, Any]:
    n = FIG5_6_EPISODES
    full = spikes.size >= n and energies.size >= n
    es = energies if is_paper else centered_smooth(energies, FIG4_SMOOTH)
    e_stats = _curve_stats(energies, es)
    e0 = _wmean(energies, 0, 50)
    m = {
        "spike_trend": _ratio(_wmean(spikes, 350, n), _wmean(spikes, 0, 50)),
        "energy_t50": e_stats["t50"] if e_stats["end_smoothed"] > e_stats["start_smoothed"] else None,
        "energy_t90": e_stats["t90"] if e_stats["end_smoothed"] > e_stats["start_smoothed"] else None,
        "energy_mid_ramp": _ratio(_wmean(energies, 55, 76), e0),
        "energy_trend": _ratio(_wmean(energies, 350, n), e0),
        "energy_overshoot": _ratio(_wmean(energies, 150, 350), _wmean(energies, 350, n)),
    }
    if is_paper:
        m["energy_pearson_0_500"] = 1.0
    else:
        paper_e = _paper_grid("fig5_energy", "Smoothed", "Raw")
        m["energy_pearson_0_500"] = _pearson(es[:n], paper_e) if energies.size >= n else float("nan")
    if not full:
        m.update({k: float("nan") for k in ("spike_trend", "energy_trend", "energy_overshoot")})
    return m


@functools.lru_cache(maxsize=1)
def _fig5_paper_cached() -> tuple[tuple[str, Any], ...]:
    spikes = _paper_grid("fig5_spikes", "Spike Count", "Smoothed", "Raw")
    energies = _paper_grid("fig5_energy", "Smoothed", "Raw")
    return tuple(_fig5_metrics(spikes, energies, is_paper=True).items())


def fig5_paper_metrics() -> dict[str, Any]:
    """Paper values of every ``FIG5_CHECKS`` metric."""
    return dict(_fig5_paper_cached())


def fig5_spikes_energy_gates(
    episode_spikes: list[float] | np.ndarray,
    episode_energies: list[float] | np.ndarray,
    *,
    early_hi: float = 50.0,
    mid_lo: float = 55.0,
    mid_hi: float = 75.0,
    late_lo: float = 350.0,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 5 spike count + DBS energy vs digitized paper curves (``FIG5_CHECKS``).

    Required rows are scale-free (shape, timing, ratios). Absolute spike and
    energy levels are report-only (pending maintainer decision; docs/figures/nguyen/5.md).
    """
    spikes = np.asarray(episode_spikes, dtype=float)
    energies = np.asarray(episode_energies, dtype=float)
    n = int(spikes.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})

    ours = _fig5_metrics(spikes, energies)
    paper = fig5_paper_metrics()
    gates = apply_checks(FIG5_CHECKS, ours, paper)

    # Report-only absolute levels (ep windows inclusive, as before).
    x = np.arange(n, dtype=float)
    psx, psy = _pick_series(load_curves("fig5_spikes"), "Spike Count", "Smoothed", "Raw")
    pex, pey = _pick_series(load_curves("fig5_energy"), "Smoothed", "Raw")
    lv: dict[str, float] = {}
    for name, (ox, oy, px, py) in {"spike": (x, spikes, psx, psy), "energy": (x, energies, pex, pey)}.items():
        lv[f"{name}_early"] = window_mean(ox, oy, hi=early_hi)
        lv[f"{name}_mid"] = window_mean(ox, oy, lo=mid_lo, hi=mid_hi)
        lv[f"{name}_late"] = window_mean(ox, oy, lo=late_lo)
        lv[f"{name}_mean"] = float(np.mean(oy))
        lv[f"paper_{name}_early"] = window_mean(px, py, hi=early_hi)
        lv[f"paper_{name}_mid"] = window_mean(px, py, lo=mid_lo, hi=mid_hi)
        lv[f"paper_{name}_late"] = window_mean(px, py, lo=late_lo)
        lv[f"paper_{name}_mean"] = float(np.mean(py))
    report = {
        f"{name}_{w}_near_paper": rel_close(lv[f"{name}_{w}"], lv[f"paper_{name}_{w}"], tol=rel_tol)
        for name in ("spike", "energy")
        for w in ("early", "mid", "late", "mean")
    }
    report["spike_stays_near_800"] = all(600.0 <= lv[f"spike_{w}"] <= 1000.0 for w in ("early", "mid", "late", "mean"))
    report["energy_monotonic_rise"] = bool(lv["energy_early"] < lv["energy_mid"] < lv["energy_late"])
    report["energy_late_above_early"] = bool(lv["energy_late"] > lv["energy_early"] * 1.25)
    report["spike_series_has_variance"] = float(np.std(spikes)) > 0.0
    report["energy_not_constant"] = float(np.std(energies)) > 0.01 * max(abs(lv["energy_mean"]), 1.0)

    metrics = {
        **ours,
        **{f"paper_{k}": v for k, v in paper.items()},
        **lv,
        "pearson_spikes": pearson_on_ref_x(psx, psy, x, spikes),
    }
    return _with_report(_gate_pack(
        gates,
        metrics,
        paper_ref={"spikes": str(curves_path("fig5_spikes")), "energy": str(curves_path("fig5_energy"))},
        notes=[
            "Required rows are scale-free; absolute spike / energy levels are report-only.",
            "Energy t50/t90: smoothed progress from ep 0–50 to ep 150–500 (paper 73 / 88).",
        ],
    ), report)


def _fig6_ab_metrics(ab: np.ndarray, *, paper_ab: np.ndarray, is_paper: bool = False) -> dict[str, Any]:
    n = FIG5_6_EPISODES
    first = int(np.flatnonzero(np.isfinite(paper_ab))[0])
    full = ab.size >= n
    smooth = ab if is_paper else centered_smooth(ab, FIG4_SMOOTH)
    early = _nan_wmean(ab, first, 50)
    mid = _nan_wmean(ab, 50, 100)
    late = _nan_wmean(ab, FIG6_LATE_LO, n)
    m = {
        "ab_first_episode": first,
        "ab_early": early,
        "ab_mid": mid,
        "ab_post100": _nan_wmean(ab, 100, 250),
        "ab_late": late if full else float("nan"),
        "ab_mean": _nan_wmean(ab, first, n) if full else float("nan"),
        "ab_t50": _milestone(smooth, (first, 50), (150, n), 0.5),
        "ab_drop_fraction": _ratio(early - mid, early - late) if full else float("nan"),
        "ab_trend": _ratio(late, early) if full else float("nan"),
        "ab_drop": (early - late) if full else float("nan"),
        "ab_pearson": 1.0 if is_paper else (
            _pearson_finite(smooth[first:n], paper_ab[first:n]) if full else float("nan")
        ),
    }
    return m


def _fig6_freq_metrics(freq: np.ndarray, *, paper_freq: np.ndarray, is_paper: bool = False) -> dict[str, Any]:
    n = FIG5_6_EPISODES
    full = freq.size >= n
    smooth = freq if is_paper else centered_smooth(freq, FIG4_SMOOTH)
    return {
        "freq_early": _nan_wmean(freq, 0, 50),
        "freq_mid": _nan_wmean(freq, 50, 100),
        "freq_late": _nan_wmean(freq, FIG6_LATE_LO, n) if full else float("nan"),
        "freq_t50": _milestone(smooth, FIG6_FREQ_START_WINDOW, (150, n), 0.5),
        "freq_pearson": 1.0 if is_paper else (_pearson(smooth[:n], paper_freq) if full else float("nan")),
    }


def _raw_points_late_cv(stem: str) -> float:
    x, y = _pick_series(load_curves(stem), "Raw")
    seg = y[x >= FIG6_LATE_LO]
    return float(np.std(seg) / np.mean(seg))


@functools.lru_cache(maxsize=1)
def _fig6_paper_cached() -> tuple[tuple[str, Any], ...]:
    ab = _paper_grid("fig6_power", "Smoothed", traced=True)
    freq = _paper_grid("fig6_freq", "Smoothed")
    m = {
        **_fig6_ab_metrics(ab, paper_ab=ab, is_paper=True),
        **_fig6_freq_metrics(freq, paper_freq=freq, is_paper=True),
        # Episode-to-episode spread: the Raw trace's vertices past ep 350.
        "ab_late_cv": _raw_points_late_cv("fig6_power"),
        "freq_late_cv": _raw_points_late_cv("fig6_freq"),
    }
    return tuple(m.items())


def fig6_paper_metrics() -> dict[str, Any]:
    """Paper values of every ``FIG6_CHECKS`` metric."""
    return dict(_fig6_paper_cached())


def fig6_power_gates(
    episode_alpha_beta: list[float] | np.ndarray,
    *,
    theta: float = THETA,
) -> dict[str, Any]:
    """Fig 6 panel (a) GPi α–β (episode-final reading) vs digitized paper curve."""
    ab = np.asarray(episode_alpha_beta, dtype=float)
    n = int(ab.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})
    paper_ab = _paper_grid("fig6_power", "Smoothed", traced=True)
    paper = fig6_paper_metrics()
    ours = {**_fig6_ab_metrics(ab, paper_ab=paper_ab), "ab_late_cv": _late_cv(ab)}
    gates = {
        "alpha_beta_early_above_theta": bool(np.isfinite(ours["ab_early"]) and ours["ab_early"] > theta),
        "alpha_beta_post100_below_theta": bool(np.isfinite(ours["ab_post100"]) and ours["ab_post100"] <= theta),
        "alpha_beta_late_below_theta": bool(np.isfinite(ours["ab_late"]) and ours["ab_late"] <= theta),
        **apply_checks(FIG6_AB_CHECKS, ours, paper),
    }
    early, mid, late = ours["ab_early"], ours["ab_mid"], ours["ab_late"]
    late_std = float(np.std(ab[FIG6_LATE_LO:])) if n > FIG6_LATE_LO else float("nan")
    report = {
        "alpha_beta_drops_by_100": bool(mid < early - 15.0),
        "alpha_beta_monotonic_drop": bool(early > mid > late),
        "alpha_beta_series_has_variance": bool(float(np.std(ab)) > 0.0),
        "alpha_beta_late_stable": bool(late_std <= 0.35 * max(abs(late), 1.0)),
        "alpha_beta_decreases_like_paper": bool(late < early),
    }
    metrics = {**ours, **{f"paper_{k}": v for k, v in paper.items() if k.startswith("ab_")}}
    return _with_report(_gate_pack(
        gates,
        metrics,
        paper_ref={"power": str(curves_path("fig6_power"))},
        notes=["Fig 6a: GPi α–β (7–35 Hz) at each episode's end over 500 training episodes."],
    ), report)


def fig6_training_gates(
    episode_alpha_beta: list[float] | np.ndarray,
    episode_amplitudes: list[float] | np.ndarray,
    episode_frequencies: list[float] | np.ndarray,
    episode_pulse_widths: list[float] | np.ndarray,
    *,
    late_lo: float = 350.0,
    late_stable_n: int = 50,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Fig 6 α–β + DBS parameters over training vs digitized paper curves (``FIG6_CHECKS``)."""
    ab = np.asarray(episode_alpha_beta, dtype=float)
    amp = np.asarray(episode_amplitudes, dtype=float)
    freq = np.asarray(episode_frequencies, dtype=float)
    pw = np.asarray(episode_pulse_widths, dtype=float)
    n = int(ab.size)
    if n < 10:
        return _gate_pack({"enough_episodes": False}, {"n_episodes": n})

    power = fig6_power_gates(ab)
    paper = fig6_paper_metrics()
    paper_freq = _paper_grid("fig6_freq", "Smoothed")
    ours_f = {**_fig6_freq_metrics(freq, paper_freq=paper_freq), "freq_late_cv": _late_cv(freq)}
    freq_gates = apply_checks(FIG6_FREQ_CHECKS, ours_f, paper)

    x = np.arange(n, dtype=float)
    amp_late = window_mean(x, amp, lo=late_lo)
    freq_late = window_mean(x, freq, lo=late_lo)
    pw_late = window_mean(x, pw, lo=late_lo)
    stable_start = max(0, n - late_stable_n)
    stds = [float(np.std(a[stable_start:])) for a in (amp, freq, pw)]
    pampx, pampy = _pick_series(load_curves("fig6_amp"), "Smoothed", "Raw")
    ppwx, ppwy = _pick_series(load_curves("fig6_pw"), "Smoothed", "Raw")
    p_amp_late = window_mean(pampx, pampy, lo=late_lo)
    p_pw_late = window_mean(ppwx, ppwy, lo=late_lo)
    amp_left = abs(amp_late - INIT_AMP) / INIT_AMP > 0.05
    freq_left = abs(freq_late - INIT_FREQ) / INIT_FREQ > 0.05
    pw_left = abs(pw_late - INIT_PW) / max(INIT_PW, 1e-9) > 0.05
    # Report-only (pending maintainer decision): amplitude and pulse width sit far above the
    # plant's charge threshold in every reachable state, so where they settle does not move
    # α–β (docs/figures/nguyen/6.md). Old frequency heuristics are superseded by FIG6_FREQ_CHECKS.
    report = {
        **power.get("report", {}),
        "amp_left_init": amp_left,
        "pw_left_init": pw_left,
        "params_left_init": bool(amp_left and freq_left and pw_left),
        "amp_late_near_paper": rel_close(amp_late, p_amp_late, tol=rel_tol),
        "pw_late_near_paper": rel_close(pw_late, p_pw_late, tol=rel_tol),
        "late_params_stable": all(
            s <= 0.20 * max(abs(m), 1e-9) for s, m in zip(stds, (amp_late, freq_late, pw_late))
        ),
        "freq_left_init": freq_left,
        "freq_late_stable": stds[1] <= 0.20 * max(abs(freq_late), 1e-9),
    }
    metrics = {
        **power["metrics"],
        **ours_f,
        **{f"paper_{k}": v for k, v in paper.items() if k.startswith("freq_")},
        "amp_late": amp_late,
        "pw_late": pw_late,
        "paper_amp_late": p_amp_late,
        "paper_pw_late": p_pw_late,
    }
    return _with_report(_gate_pack(
        {**power["gates"], **freq_gates},
        metrics,
        paper_ref={
            "power": str(curves_path("fig6_power")),
            "amp": str(curves_path("fig6_amp")),
            "freq": str(curves_path("fig6_freq")),
            "pw": str(curves_path("fig6_pw")),
        },
        notes=[
            "α–β and frequency rows are required and calibrated; amplitude / pulse width are report-only.",
            "Fig 6a: GPi α–β (7–35 Hz) at each episode's end over 500 training episodes.",
        ],
    ), report)


# Paper §IV: "We evaluate our trained model against 50 test episodes".
PAPER_FIG7_EPISODES = 50
FIG7_STEPS = 26  # reset reading + 25 steps (band traced at steps 0–25)
FIG7_BAND_Z = 1.96  # the shaded band reads as ±1.96 SD of single episodes (docs/figures/nguyen/7.md)
FIG7_PEARSON_MIN = 0.80  # paper shifted one step: r = 0.77
FIG7_PEAK_STEP_SLACK = 1
FIG7_WINDOWS: dict[str, tuple[int, int]] = {  # inclusive step ranges
    "early": (0, 5),
    "mid": (6, 15),
    "late": (18, 25),
    "overall": (0, 25),
}


@functools.lru_cache(maxsize=1)
def _fig7_paper_cached() -> dict[str, Any]:
    paper = load_curves("fig7")
    pax, pay = _pick_series(paper, "average", "Smoothed", "Raw")
    steps = np.arange(FIG7_STEPS, dtype=float)
    avg = np.interp(steps, pax, pay)
    upper = np.interp(steps, *paper["ci_upper"])
    lower = np.interp(steps, *paper["ci_lower"])
    sd = (upper - lower) / (2.0 * FIG7_BAND_Z)
    peak = float(np.max(pay))
    peak_step = int(round(float(pax[np.argmax(pay)])))
    win = {k: float(np.mean(avg[a : b + 1])) for k, (a, b) in FIG7_WINDOWS.items()}
    win_sd = {k: float(np.mean(sd[a : b + 1])) for k, (a, b) in FIG7_WINDOWS.items()}
    late = win["late"]
    sd_peak, sd_late = float(sd[peak_step]), win_sd["late"]
    values = {
        "start": float(avg[0]),
        "peak": peak,
        **{f"{k}_mean": v for k, v in win.items()},
        "drop": peak - late,
        "late_peak_ratio": late / peak,
    }
    tols = {
        "start": float(sd[0]),
        "peak": sd_peak,
        **{f"{k}_mean": v for k, v in win_sd.items()},
        "drop": float(np.hypot(sd_peak, sd_late)),
        "late_peak_ratio": float((late / peak) * np.hypot(sd_late / late, sd_peak / peak)),
    }
    return {
        "x": pax, "y": pay, "avg": avg, "sd": sd, "values": values, "tols": tols,
        "peak_x": float(pax[np.argmax(pay)]), "peak_step": peak_step,
        "n_points": int(round(max(float(np.max(paper["ci_upper"][0])), float(np.max(paper["ci_lower"][0]))))) + 1,
    }


def fig7_checks() -> tuple[Check, ...]:
    """Fig 7 checks; each tolerance is the paper's single-episode SD (band half-width / 1.96)."""
    t = _fig7_paper_cached()["tols"]
    return (
        Check("start_near_paper", "start", "abs", t["start"], "step 0; SD at step 0"),
        Check("peak_power_near_paper", "peak", "abs", t["peak"], "max of mean trace; SD at paper peak step"),
        Check("early_mean_near_paper", "early_mean", "abs", t["early_mean"], "steps 0–5; mean SD"),
        Check("mid_mean_near_paper", "mid_mean", "abs", t["mid_mean"], "steps 6–15; mean SD"),
        Check("late_mean_near_paper", "late_mean", "abs", t["late_mean"], "steps 18–25; mean SD"),
        Check("overall_mean_near_paper", "overall_mean", "abs", t["overall_mean"], "steps 0–25; mean SD"),
        Check("peak_to_late_drop", "drop", "abs", t["drop"], "peak − late; SDs in quadrature"),
        Check("late_peak_ratio_near_paper", "late_peak_ratio", "abs", t["late_peak_ratio"], "late / peak; propagated SD"),
    )


def fig7_paper_metrics() -> dict[str, Any]:
    return dict(_fig7_paper_cached()["values"])


def _fig7_trace_metrics(mean_trace: np.ndarray) -> dict[str, Any]:
    m: dict[str, Any] = {"start": float(mean_trace[0]) if mean_trace.size else float("nan")}
    if mean_trace.size and np.any(np.isfinite(mean_trace)):
        m["peak"] = float(np.nanmax(mean_trace))
        m["peak_step"] = int(np.nanargmax(mean_trace))
    else:
        m["peak"], m["peak_step"] = float("nan"), -1
    for k, (a, b) in FIG7_WINDOWS.items():
        seg = mean_trace[a : b + 1]
        m[f"{k}_mean"] = float(np.nanmean(seg)) if seg.size and np.any(np.isfinite(seg)) else float("nan")
    m["drop"] = m["peak"] - m["late_mean"]
    m["late_peak_ratio"] = _ratio(m["late_mean"], m["peak"])
    return m


def fig7_eval_gates(
    alpha_beta_trajectories: list[list[float]],
    *,
    fig3_pd_on_median: float | None = None,
) -> dict[str, Any]:
    """Fig 7 eval mean α–β trace vs the digitized paper average curve and band.

    Pass the trajectories as recorded (unpadded): the protocol gate checks that
    every episode covers the paper's full step range.
    """
    if not alpha_beta_trajectories:
        return _gate_pack({"trajectories_present": False}, {})

    max_len = max(len(tr) for tr in alpha_beta_trajectories)
    min_len = min(len(tr) for tr in alpha_beta_trajectories)
    step_means: list[float] = []
    for step in range(max_len):
        vals = [float(tr[step]) for tr in alpha_beta_trajectories if step < len(tr)]
        step_means.append(float(np.mean(vals)) if vals else float("nan"))
    steps = np.arange(len(step_means), dtype=float)
    mean_trace = np.asarray(step_means, dtype=float)

    ref = _fig7_paper_cached()
    paper = fig7_paper_metrics()
    ours = _fig7_trace_metrics(mean_trace)
    checks = fig7_checks()
    r_shape = pearson_on_ref_x(ref["x"], ref["y"], steps, mean_trace)

    gates: dict[str, bool] = {
        "eval_protocol_ok": bool(
            len(alpha_beta_trajectories) >= PAPER_FIG7_EPISODES and min_len >= ref["n_points"]
        ),
        "step_series_finite": bool(np.isfinite(mean_trace).all()),
        "peak_step_timing": abs(ours["peak_step"] - ref["peak_step"]) <= FIG7_PEAK_STEP_SLACK,
        **apply_checks(checks, ours, paper),
        "pearson_shape_ok": bool(np.isfinite(r_shape) and r_shape >= FIG7_PEARSON_MIN),
    }
    if fig3_pd_on_median is not None and np.isfinite(fig3_pd_on_median):
        # Below both our Fig 3 PD On median and the paper's PD On readout.
        gates["below_fig3_pd_median"] = ours["overall_mean"] < min(
            float(fig3_pd_on_median), PAPER_FIG3_PD_ON_MEAN
        )
    metrics: dict[str, Any] = {
        **{f"{k}_val" if k in ("start", "peak") else k: v for k, v in ours.items()},
        **{f"paper_{k}_val" if k in ("start", "peak") else f"paper_{k}": v for k, v in paper.items()},
        **{f"tol_{c.metric}": c.tol for c in checks},
        "paper_peak_step": ref["peak_x"],
        "paper_peak_step_rounded": ref["peak_step"],
        "n_episodes": len(alpha_beta_trajectories),
        "paper_n_episodes": PAPER_FIG7_EPISODES,
        "n_steps": max_len,
        "min_steps": min_len,
        "paper_n_steps": ref["n_points"],
        "pearson_mean_trace": r_shape,
        "mean_below_theta": ours["overall_mean"] <= THETA,
        "paper_mean_below_theta": paper["overall_mean"] <= THETA,
    }
    return _gate_pack(
        gates,
        metrics,
        paper_ref={"path": str(curves_path("fig7"))},
        notes=[
            "Fig 7 uses the per-step mean across eval episodes vs the paper average curve.",
            "Window tolerances = the paper's single-episode SD (band half-width / 1.96).",
        ],
    )
