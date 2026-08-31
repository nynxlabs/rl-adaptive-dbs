"""Digitization-anchored gate helpers for Ravivarapu (SEA-DBS) panels.

Gate rules mirror ``docs/figures/papers/ravivarapu.md`` — ordering, shared
starts, relative drops, and cross-panel ratios. Absolute y uses generous bands
because plant scale may differ from paper crops.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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

ARTIFACT_ROOT = Path("artifacts/figures/papers/ravivarapu/paper_digitization")


def curves_path(stem: str) -> Path:
    return ARTIFACT_ROOT / f"curves_{stem}.json"


def load_curves(stem: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from normalized ``curves_{stem}.json``."""
    return load_refined(curves_path(stem))


def attach_digitization(
    heuristic: dict[str, Any],
    dig_report: dict[str, Any],
    *,
    prefix: str = "dig_",
) -> dict[str, Any]:
    """Merge heuristic gate dict with digitization report; ``pass`` = both."""
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


def merge_gate_report(dig: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``_gate_pack`` output for panel manifests."""
    out: dict[str, Any] = {str(k): bool(v) for k, v in dig.get("gates", {}).items()}
    out.update(extra)
    out["pass"] = bool(dig.get("pass", False))
    metrics = dig.get("metrics")
    if metrics:
        out["gate_metrics"] = metrics
    notes = dig.get("notes")
    if notes:
        out["gate_notes"] = list(notes)
    return out


# Fig 4a gate tiers: shape = ordering + trajectory profile; full adds digitization polish.
RAVIVARAPU_FIG4A_GATE_TIER: dict[str, str] = {
    # Shape tier (16 gates)
    "n_episodes_ok": "shape",
    "shared_start": "shape",
    "baseline_declines": "shape",
    "sea_declines": "shape",
    "sea_below_baseline_mid": "shape",
    "sea_below_baseline_midlate": "shape",
    "sea_below_baseline_late": "shape",
    "sea_steeper_drop_than_baseline": "shape",
    "gap_widens_over_training": "shape",
    "late_gap_substantial": "shape",
    "drop_timing_baseline": "shape",
    "drop_timing_sea": "shape",
    "gradual_decline_baseline": "shape",
    "gradual_decline_sea": "shape",
    "pearson_baseline_min": "shape",
    "pearson_sea_min": "shape",
    # Full tier (11 gates)
    "shared_start_near_paper": "full",
    "baseline_drop_vs_paper": "full",
    "sea_drop_vs_paper": "full",
    "sea_mid_near_paper": "full",
    "sea_midlate_near_paper": "full",
    "sea_late_near_paper": "full",
    "mid_gap_near_paper": "full",
    "midlate_gap_near_paper": "full",
    "late_gap_near_paper": "full",
    "late_early_ratio_baseline_near_paper": "full",
    "late_early_ratio_sea_near_paper": "full",
}


# Fig 4b gate tiers: shape = ordering + trajectory profile; full adds digitization polish.
RAVIVARAPU_FIG4B_GATE_TIER: dict[str, str] = {
    # Shape tier (14 gates)
    "n_episodes_ok": "shape",
    "shared_start": "shape",
    "baseline_rises": "shape",
    "sea_rises": "shape",
    "sea_above_baseline_late": "shape",
    "sea_steeper_rise_than_baseline": "shape",
    "paper_pull_ahead_mid": "shape",
    "late_gap_substantial": "shape",
    "rise_timing_baseline": "shape",
    "rise_timing_sea": "shape",
    "gradual_rise_baseline": "shape",
    "gradual_rise_sea": "shape",
    "pearson_baseline_min": "shape",
    "pearson_sea_min": "shape",
    # Full tier (5 gates)
    "shared_start_near_paper": "full",
    "baseline_rise_vs_paper": "full",
    "sea_rise_vs_paper": "full",
    "late_gap_near_paper": "full",
    "midlate_gap_near_paper": "full",
}


def ravivarapu_fig4a_tier_pass(flat: dict[str, Any], *, full: bool) -> bool:
    """Return whether every gate in the shape or full tier passes."""
    tiers = ("shape", "full") if full else ("shape",)
    gates_dict = flat.get("gates", flat)
    for key, tier in RAVIVARAPU_FIG4A_GATE_TIER.items():
        if tier not in tiers:
            continue
        value = gates_dict.get(key)
        if isinstance(value, bool) and not value:
            return False
    return True


def ravivarapu_fig4a_attach_tiered_pass(flat: dict[str, Any]) -> dict[str, Any]:
    """Set ``shape_pass`` (phase 1) and ``pass`` (ship) on a flattened Fig 4a manifest."""
    out = dict(flat)
    out["shape_pass"] = ravivarapu_fig4a_tier_pass(out, full=False)
    out["pass"] = ravivarapu_fig4a_tier_pass(out, full=True)
    return out


def ravivarapu_fig4b_tier_pass(flat: dict[str, Any], *, full: bool) -> bool:
    """Return whether every gate in the shape or full tier passes for Fig 4b."""
    tiers = ("shape", "full") if full else ("shape",)
    gates_dict = flat.get("gates", flat)
    for key, tier in RAVIVARAPU_FIG4B_GATE_TIER.items():
        if tier not in tiers:
            continue
        value = gates_dict.get(key)
        if isinstance(value, bool) and not value:
            return False
    return True


def ravivarapu_fig4b_attach_tiered_pass(flat: dict[str, Any]) -> dict[str, Any]:
    """Set ``shape_pass`` (phase 1) and ``pass`` (ship) on a flattened Fig 4b manifest."""
    out = dict(flat)
    out["shape_pass"] = ravivarapu_fig4b_tier_pass(out, full=False)
    out["pass"] = ravivarapu_fig4b_tier_pass(out, full=True)
    return out


def _as_fy(y: Sequence[float]) -> np.ndarray:
    return np.asarray(y, dtype=float)


def _early_late(
    y: Sequence[float],
    *,
    early_frac: float = 0.05,
    late_frac: float = 0.5,
    min_early: int = 3,
) -> tuple[float, float]:
    arr = _as_fy(y)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    early_n = max(min_early, int(n * early_frac))
    late_start = max(early_n, int(n * (1.0 - late_frac)))
    return float(np.mean(arr[:early_n])), float(np.mean(arr[late_start:]))


def _mid_mean(y: Sequence[float], lo: int, hi: int) -> float:
    arr = _as_fy(y)
    if arr.size == 0:
        return float("nan")
    lo_i = max(0, lo)
    hi_i = min(int(arr.size), hi)
    if hi_i <= lo_i:
        return float("nan")
    return float(np.mean(arr[lo_i:hi_i]))


def _mean_first_n(y: Sequence[float], n: int = 15) -> float:
    arr = _as_fy(y)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr[: min(n, int(arr.size))]))


def _drop_fraction_by_episode(
    x: np.ndarray,
    y: np.ndarray,
    *,
    episode: float,
    early_hi: float,
    late_lo: float,
) -> float:
    """Share of early→late drop achieved by ``episode`` (exclusive early window)."""
    early = window_mean(x, y, hi=early_hi)
    late = window_mean(x, y, lo=late_lo)
    total = early - late
    if not np.isfinite(total) or total <= 1e-9:
        return float("nan")
    mid_hi = max(early_hi, episode)
    mid = window_mean(x, y, lo=early_hi, hi=mid_hi)
    if not np.isfinite(mid):
        return float("nan")
    return float((early - mid) / total)


def _rise_fraction_by_episode(
    x: np.ndarray,
    y: np.ndarray,
    *,
    episode: float,
    early_hi: float,
    late_lo: float,
) -> float:
    """Share of early→late rise achieved by ``episode`` (exclusive early window)."""
    early = window_mean(x, y, hi=early_hi)
    late = window_mean(x, y, lo=late_lo)
    total = late - early
    if not np.isfinite(total) or total <= 1e-9:
        return float("nan")
    mid_hi = max(early_hi, episode)
    mid = window_mean(x, y, lo=early_hi, hi=mid_hi)
    if not np.isfinite(mid):
        return float("nan")
    return float((mid - early) / total)


def _drop_timing_ok(
    ours_frac: float,
    paper_frac: float,
    *,
    slack_mult: float = 1.55,
    slack_abs: float = 0.10,
) -> bool:
    """True when our cumulative drop by a checkpoint is not front-loaded vs paper."""
    if not np.isfinite(ours_frac) or not np.isfinite(paper_frac):
        return False
    cap = max(paper_frac * slack_mult, paper_frac + slack_abs)
    return ours_frac <= cap


def _nonincreasing_window_means(*values: float, flat_tol: float = 0.003) -> bool:
    """True when each window mean is at or below the prior (paper Fig 4a glide-down)."""
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) < 2:
        return False
    return all(vals[i + 1] <= vals[i] + flat_tol for i in range(len(vals) - 1))


def _nondecreasing_gaps(*gaps: float, flat_tol: float = 0.003) -> bool:
    """True when baseline−SEA separation never closes across late windows."""
    vals = [float(g) for g in gaps if np.isfinite(g)]
    if len(vals) < 2:
        return False
    return all(vals[i + 1] + flat_tol >= vals[i] for i in range(len(vals) - 1))


def _interp_at(x: np.ndarray, y: np.ndarray, episode: float) -> float:
    """Interpolate a digitized/replication curve at one episode."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.interp(float(episode), x, y))


def ravivarapu_fig4a_gates(
    baseline_psd: Sequence[float],
    sea_psd: Sequence[float],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    early_hi: float = 15.0,
    mid_lo: float = 40.0,
    mid_hi: float = 80.0,
    late_lo: float = 120.0,
    timing_episode: float = 50.0,
    drop_frac_of_paper: float = 0.40,
    profile_frac_of_paper: float = 0.30,
    pearson_baseline_min: float = 0.55,
    pearson_sea_min: float = 0.55,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    n_expected: int = 150,
    shared_start_tol: float = 0.05,
    late_gap_min: float = 0.02,
) -> dict[str, Any]:
    """Unified Fig 4a episode PSD gates vs digitized paper learning curves.

    Shape tier: 150 episodes, shared untreated start, genuine learning drops
    on both variants, SEA steeper and lower than Baseline late, clear late gap,
    not front-loaded drop by ep 50, gradual mid-to-late decline, and Pearson r.
    Full tier: digitization scale polish (shared start, drops vs paper, late gap,
    late/early ratios).
    """
    paper = paper or load_curves("fig4a")
    pbx, pby = paper["Baseline"]
    psx, psy = paper["SEA-DBS"]

    n = min(len(baseline_psd), len(sea_psd))
    if n < 10:
        return _gate_pack(
            {"n_episodes_ok": False},
            {"n_episodes": n},
            paper_ref={"path": str(curves_path("fig4a"))},
        )

    x = np.arange(n, dtype=float)
    b = _as_fy(baseline_psd[:n])
    s = _as_fy(sea_psd[:n])

    b_early = window_mean(x, b, hi=early_hi)
    b_mid = window_mean(x, b, lo=mid_lo, hi=mid_hi)
    b_midlate_w = window_mean(x, b, lo=70.0, hi=110.0)
    b_late = window_mean(x, b, lo=late_lo)

    s_early = window_mean(x, s, hi=early_hi)
    s_mid = window_mean(x, s, lo=mid_lo, hi=mid_hi)
    s_midlate_w = window_mean(x, s, lo=70.0, hi=110.0)
    s_late = window_mean(x, s, lo=late_lo)

    pb_early = window_mean(pbx, pby, hi=early_hi)
    pb_mid = window_mean(pbx, pby, lo=mid_lo, hi=mid_hi)
    pb_midlate_w = window_mean(pbx, pby, lo=70.0, hi=110.0)
    pb_late = window_mean(pbx, pby, lo=late_lo)

    ps_early = window_mean(psx, psy, hi=early_hi)
    ps_mid = window_mean(psx, psy, lo=mid_lo, hi=mid_hi)
    ps_midlate_w = window_mean(psx, psy, lo=70.0, hi=110.0)
    ps_late = window_mean(psx, psy, lo=late_lo)

    b_drop = b_early - b_late
    s_drop = s_early - s_late
    pb_drop = pb_early - pb_late
    ps_drop = ps_early - ps_late

    b_midlate = b_mid - b_late
    s_midlate = s_mid - s_late
    pb_midlate = pb_mid - pb_late
    ps_midlate = ps_mid - ps_late

    b_drop_frac = _drop_fraction_by_episode(
        x, b, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    s_drop_frac = _drop_fraction_by_episode(
        x, s, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    pb_drop_frac = _drop_fraction_by_episode(
        pbx, pby, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    ps_drop_frac = _drop_fraction_by_episode(
        psx, psy, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )

    shape_b = pearson_on_ref_x(pbx, pby, x, b)
    shape_s = pearson_on_ref_x(psx, psy, x, s)

    gap_early = b_early - s_early
    p_gap_early = pb_early - ps_early
    gap_mid = b_mid - s_mid
    p_gap_mid = pb_mid - ps_mid
    gap_midlate = b_midlate_w - s_midlate_w
    p_gap_midlate = pb_midlate_w - ps_midlate_w
    gap = b_late - s_late
    p_gap = pb_late - ps_late

    gates = {
        # Shape tier
        "n_episodes_ok": n >= n_expected,
        "shared_start": bool(
            np.isfinite(b_early)
            and np.isfinite(s_early)
            and abs(b_early - s_early) <= shared_start_tol
        ),
        "baseline_declines": bool(np.isfinite(b_drop) and b_drop >= 0.02),
        "sea_declines": bool(np.isfinite(s_drop) and s_drop >= 0.04),
        "sea_below_baseline_mid": bool(
            np.isfinite(s_mid) and np.isfinite(b_mid) and s_mid < b_mid
        ),
        "sea_below_baseline_midlate": bool(
            np.isfinite(s_midlate_w) and np.isfinite(b_midlate_w) and s_midlate_w < b_midlate_w
        ),
        "sea_below_baseline_late": bool(
            np.isfinite(s_late) and np.isfinite(b_late) and s_late < b_late
        ),
        "sea_steeper_drop_than_baseline": bool(
            np.isfinite(s_drop) and np.isfinite(b_drop) and s_drop > b_drop
        ),
        "gap_widens_over_training": bool(
            np.isfinite(gap_mid)
            and np.isfinite(gap_midlate)
            and np.isfinite(gap)
            and gap_mid >= -0.005
            and gap_midlate >= gap_mid - 0.005
            and gap >= gap_midlate - 0.008
        ),
        "late_gap_substantial": bool(
            np.isfinite(gap) and 0.015 <= gap <= 0.065
        ),
        "drop_timing_baseline": _drop_timing_ok(b_drop_frac, pb_drop_frac),
        "drop_timing_sea": _drop_timing_ok(s_drop_frac, ps_drop_frac),
        "gradual_decline_baseline": bool(
            np.isfinite(b_midlate)
            and np.isfinite(pb_midlate)
            and b_midlate >= profile_frac_of_paper * pb_midlate
        ),
        "gradual_decline_sea": bool(
            np.isfinite(s_midlate)
            and np.isfinite(ps_midlate)
            and s_midlate >= profile_frac_of_paper * ps_midlate
        ),
        "pearson_baseline_min": bool(
            np.isfinite(shape_b) and shape_b >= pearson_baseline_min
        ),
        "pearson_sea_min": bool(np.isfinite(shape_s) and shape_s >= pearson_sea_min),
        # Full tier
        "shared_start_near_paper": rel_close(s_early, ps_early, tol=rel_tol)
        and rel_close(b_early, pb_early, tol=rel_tol),
        "baseline_drop_vs_paper": bool(
            np.isfinite(b_drop) and np.isfinite(pb_drop) and b_drop >= drop_frac_of_paper * pb_drop
        ),
        "sea_drop_vs_paper": bool(
            np.isfinite(s_drop) and np.isfinite(ps_drop) and s_drop >= drop_frac_of_paper * ps_drop
        ),
        "sea_mid_near_paper": rel_close(s_mid, ps_mid, tol=0.15),
        "sea_midlate_near_paper": rel_close(s_midlate_w, ps_midlate_w, tol=0.15),
        "sea_late_near_paper": rel_close(s_late, ps_late, tol=0.15),
        "mid_gap_near_paper": bool(
            np.isfinite(gap_mid)
            and np.isfinite(p_gap_mid)
            and p_gap_mid > 0
            and 0.40 * p_gap_mid <= gap_mid <= 2.50 * p_gap_mid
        ),
        "midlate_gap_near_paper": bool(
            np.isfinite(gap_midlate)
            and np.isfinite(p_gap_midlate)
            and p_gap_midlate > 0
            and 0.40 * p_gap_midlate <= gap_midlate <= 2.60 * p_gap_midlate
        ),
        "late_gap_near_paper": bool(
            np.isfinite(gap)
            and np.isfinite(p_gap)
            and p_gap > 0
            and 0.40 * p_gap <= gap <= 2.20 * p_gap
        ),
        "late_early_ratio_baseline_near_paper": ratio_close(
            b_late, b_early, pb_late, pb_early, tol=ratio_tol
        ),
        "late_early_ratio_sea_near_paper": ratio_close(
            s_late, s_early, ps_late, ps_early, tol=ratio_tol
        ),
    }

    metrics = {
        "b_early": b_early,
        "b_mid": b_mid,
        "b_midlate_w": b_midlate_w,
        "b_late": b_late,
        "s_early": s_early,
        "s_mid": s_mid,
        "s_midlate_w": s_midlate_w,
        "s_late": s_late,
        "b_drop": b_drop,
        "s_drop": s_drop,
        "b_midlate": b_midlate,
        "s_midlate": s_midlate,
        "b_drop_frac_at_timing": b_drop_frac,
        "s_drop_frac_at_timing": s_drop_frac,
        "paper_b_early": pb_early,
        "paper_b_mid": pb_mid,
        "paper_b_midlate_w": pb_midlate_w,
        "paper_b_late": pb_late,
        "paper_s_early": ps_early,
        "paper_s_mid": ps_mid,
        "paper_s_midlate_w": ps_midlate_w,
        "paper_s_late": ps_late,
        "paper_b_drop": pb_drop,
        "paper_s_drop": ps_drop,
        "paper_b_midlate": pb_midlate,
        "paper_s_midlate": ps_midlate,
        "paper_b_drop_frac_at_timing": pb_drop_frac,
        "paper_s_drop_frac_at_timing": ps_drop_frac,
        "early_gap": gap_early,
        "paper_early_gap": p_gap_early,
        "mid_gap": gap_mid,
        "paper_mid_gap": p_gap_mid,
        "midlate_gap": gap_midlate,
        "paper_midlate_gap": p_gap_midlate,
        "late_gap": gap,
        "paper_late_gap": p_gap,
        "pearson_baseline": shape_b,
        "pearson_sea": shape_s,
        "pearson_baseline_min": pearson_baseline_min,
        "pearson_sea_min": pearson_sea_min,
        "timing_episode": timing_episode,
        "n_episodes": n,
    }

    return _gate_pack(
        gates,
        metrics,
        paper_ref={
            "path": str(curves_path("fig4a")),
            "early_hi": early_hi,
            "mid_lo": mid_lo,
            "mid_hi": mid_hi,
            "late_lo": late_lo,
            "timing_episode": timing_episode,
            "source": "refined/fig4a_refined.wpd.tar → curves_fig4a.json",
        },
        notes=[
            "Shape gates verify macro trajectory dynamics without over-constraining single-seed noise.",
            f"Pearson r minima: baseline>={pearson_baseline_min}, SEA>={pearson_sea_min}.",
        ],
    )


def ravivarapu_fig4a_digitization_gates(
    baseline_psd: Sequence[float],
    sea_psd: Sequence[float],
    *,
    n_expected: int = 150,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for ``ravivarapu_fig4a_gates``."""
    return ravivarapu_fig4a_gates(baseline_psd, sea_psd, n_expected=n_expected, **kwargs)


def ravivarapu_fig4b_gates(
    baseline_reward: Sequence[float],
    sea_reward: Sequence[float],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    early_hi: float = 15.0,
    mid_lo: float = 40.0,
    mid_hi: float = 80.0,
    late_lo: float = 120.0,
    timing_episode: float = 50.0,
    rise_frac_of_paper: float = 0.40,
    profile_frac_of_paper: float = 0.25,
    pearson_baseline_min: float = 0.55,
    pearson_sea_min: float = 0.55,
    n_expected: int = 150,
) -> dict[str, Any]:
    """Unified Fig 4b episode reward gates vs digitized paper learning curves.

    Shape tier: 150 episodes, shared untreated start, genuine learning rise
    on both variants, SEA steeper and higher than Baseline late, clear late gap,
    not front-loaded rise by ep 50, gradual mid-to-late increase, and Pearson r.
    Full tier: digitization scale polish (shared start, rises vs paper, late gap,
    midlate gap).
    """
    paper = paper or load_curves("fig4b")
    pbx, pby = paper["Baseline Reward"]
    psx, psy = paper["SEA-DBS Reward"]

    n = min(len(baseline_reward), len(sea_reward))
    if n < 10:
        return _gate_pack(
            {"n_episodes_ok": False},
            {"n_episodes": n},
            paper_ref={"path": str(curves_path("fig4b"))},
        )

    x = np.arange(n, dtype=float)
    b = _as_fy(baseline_reward[:n])
    s = _as_fy(sea_reward[:n])

    b_early = window_mean(x, b, hi=early_hi)
    b_mid = window_mean(x, b, lo=mid_lo, hi=mid_hi)
    b_midlate_w = window_mean(x, b, lo=70.0, hi=110.0)
    b_late = window_mean(x, b, lo=late_lo)

    s_early = window_mean(x, s, hi=early_hi)
    s_mid = window_mean(x, s, lo=mid_lo, hi=mid_hi)
    s_midlate_w = window_mean(x, s, lo=70.0, hi=110.0)
    s_late = window_mean(x, s, lo=late_lo)

    pb_early = window_mean(pbx, pby, hi=early_hi)
    pb_mid = window_mean(pbx, pby, lo=mid_lo, hi=mid_hi)
    pb_midlate_w = window_mean(pbx, pby, lo=70.0, hi=110.0)
    pb_late = window_mean(pbx, pby, lo=late_lo)

    ps_early = window_mean(psx, psy, hi=early_hi)
    ps_mid = window_mean(psx, psy, lo=mid_lo, hi=mid_hi)
    ps_midlate_w = window_mean(psx, psy, lo=70.0, hi=110.0)
    ps_late = window_mean(psx, psy, lo=late_lo)

    b_rise = b_late - b_early
    s_rise = s_late - s_early
    pb_rise = pb_late - pb_early
    ps_rise = ps_late - ps_early

    b_midlate = b_late - b_mid
    s_midlate = s_late - s_mid
    pb_midlate = pb_late - pb_mid
    ps_midlate = ps_late - ps_mid

    b_rise_frac = _rise_fraction_by_episode(
        x, b, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    s_rise_frac = _rise_fraction_by_episode(
        x, s, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    pb_rise_frac = _rise_fraction_by_episode(
        pbx, pby, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )
    ps_rise_frac = _rise_fraction_by_episode(
        psx, psy, episode=timing_episode, early_hi=early_hi, late_lo=late_lo
    )

    shape_b = pearson_on_ref_x(pbx, pby, x, b)
    shape_s = pearson_on_ref_x(psx, psy, x, s)

    gap = s_late - b_late
    gap_midlate = s_midlate_w - b_midlate_w
    p_gap = ps_late - pb_late
    p_gap_midlate = ps_midlate_w - pb_midlate_w

    scale = abs(b_early / pb_early) if abs(pb_early) > 1e-6 else 1.0
    pb_scaled_rise = pb_rise * scale
    ps_scaled_rise = ps_rise * scale
    p_gap_scaled = p_gap * scale
    p_gap_midlate_scaled = p_gap_midlate * scale

    shared_start_dist = abs(b_early - s_early) / (abs(b_early) + 1e-6)

    gates = {
        # Shape tier
        "n_episodes_ok": n >= n_expected,
        "shared_start": bool(
            np.isfinite(b_early)
            and np.isfinite(s_early)
            and shared_start_dist <= 0.25
        ),
        "baseline_rises": bool(np.isfinite(b_rise) and b_rise >= 0.25 * pb_scaled_rise),
        "sea_rises": bool(np.isfinite(s_rise) and s_rise >= 0.25 * ps_scaled_rise),
        "sea_above_baseline_late": bool(
            np.isfinite(s_late) and np.isfinite(b_late) and s_late > b_late
        ),
        "sea_steeper_rise_than_baseline": bool(
            np.isfinite(s_rise) and np.isfinite(b_rise) and s_rise > b_rise
        ),
        "paper_pull_ahead_mid": bool(
            np.isfinite(s_mid) and np.isfinite(b_mid) and s_mid > b_mid
        ),
        "late_gap_substantial": bool(np.isfinite(gap) and gap >= 0.35 * p_gap_scaled),
        "rise_timing_baseline": _drop_timing_ok(b_rise_frac, pb_rise_frac),
        "rise_timing_sea": _drop_timing_ok(s_rise_frac, ps_rise_frac),
        "gradual_rise_baseline": bool(
            np.isfinite(b_midlate)
            and np.isfinite(pb_midlate)
            and b_midlate >= profile_frac_of_paper * (pb_midlate * scale)
        ),
        "gradual_rise_sea": bool(
            np.isfinite(s_midlate)
            and np.isfinite(ps_midlate)
            and s_midlate >= profile_frac_of_paper * (ps_midlate * scale)
        ),
        "pearson_baseline_min": bool(
            np.isfinite(shape_b) and shape_b >= pearson_baseline_min
        ),
        "pearson_sea_min": bool(
            np.isfinite(shape_s) and shape_s >= pearson_sea_min
        ),
        # Full tier
        "shared_start_near_paper": rel_close(s_early, ps_early * scale, tol=0.30),
        "baseline_rise_vs_paper": bool(
            np.isfinite(b_rise)
            and np.isfinite(pb_scaled_rise)
            and b_rise >= rise_frac_of_paper * pb_scaled_rise
        ),
        "sea_rise_vs_paper": bool(
            np.isfinite(s_rise)
            and np.isfinite(ps_scaled_rise)
            and s_rise >= rise_frac_of_paper * ps_scaled_rise
        ),
        "late_gap_near_paper": bool(
            np.isfinite(gap)
            and np.isfinite(p_gap_scaled)
            and gap >= 0.50 * p_gap_scaled
        ),
        "midlate_gap_near_paper": bool(
            np.isfinite(gap_midlate)
            and np.isfinite(p_gap_midlate_scaled)
            and gap_midlate >= 0.40 * p_gap_midlate_scaled
        ),
    }

    metrics = {
        "b_early": b_early,
        "b_mid": b_mid,
        "b_midlate_w": b_midlate_w,
        "b_late": b_late,
        "s_early": s_early,
        "s_mid": s_mid,
        "s_midlate_w": s_midlate_w,
        "s_late": s_late,
        "b_rise": b_rise,
        "s_rise": s_rise,
        "b_midlate": b_midlate,
        "s_midlate": s_midlate,
        "b_rise_frac_at_timing": b_rise_frac,
        "s_rise_frac_at_timing": s_rise_frac,
        "paper_b_early": pb_early,
        "paper_b_mid": pb_mid,
        "paper_b_midlate_w": pb_midlate_w,
        "paper_b_late": pb_late,
        "paper_s_early": ps_early,
        "paper_s_mid": ps_mid,
        "paper_s_midlate_w": ps_midlate_w,
        "paper_s_late": ps_late,
        "paper_b_rise": pb_rise,
        "paper_s_rise": ps_rise,
        "paper_b_midlate": pb_midlate,
        "paper_s_midlate": ps_midlate,
        "paper_b_rise_frac_at_timing": pb_rise_frac,
        "paper_s_rise_frac_at_timing": ps_rise_frac,
        "late_gap": gap,
        "paper_late_gap": p_gap,
        "paper_late_gap_scaled": p_gap_scaled,
        "midlate_gap": gap_midlate,
        "paper_midlate_gap": p_gap_midlate,
        "paper_midlate_gap_scaled": p_gap_midlate_scaled,
        "pearson_baseline": shape_b,
        "pearson_sea": shape_s,
        "pearson_baseline_min": pearson_baseline_min,
        "pearson_sea_min": pearson_sea_min,
        "timing_episode": timing_episode,
        "n_episodes": n,
    }

    return _gate_pack(
        gates,
        metrics,
        paper_ref={
            "path": str(curves_path("fig4b")),
            "early_hi": early_hi,
            "mid_lo": mid_lo,
            "mid_hi": mid_hi,
            "late_lo": late_lo,
            "timing_episode": timing_episode,
            "source": "refined/fig4b_refined.wpd.tar → curves_fig4b.json",
        },
        notes=[
            "Shape gates verify macro trajectory dynamics without over-constraining single-seed noise.",
            f"Pearson r minima: baseline>={pearson_baseline_min}, SEA>={pearson_sea_min}.",
        ],
    )


# Fig 5 shape only: any net drop counts. Digitized paper drops (~0.10/0.15 at
# 50 Hz, ~0.06/0.07 at 30 Hz) are reference, not gate thresholds.
INFERENCE_DECLINE_MIN = 0.0
INFERENCE_SHARED_START_MAX = 0.05
# Fig 5a steps 0–5 vs digitized paper (norm = crop PSD / 1000). Generous:
# match the early window "to an extent", not Fig 4a-style magnitude polish.
INFERENCE_EARLY_N = 6
INFERENCE_EARLY_MAE_MAX = 0.030
INFERENCE_EARLY_SEA_MAE_3_5_MAX = 0.020
INFERENCE_EARLY_SEA_DROP_MIN = 0.050
# Fig 5a steps 5–10: paper keeps falling (SEA ~0.338→0.310, Baseline
# ~0.404→0.356). n_obs=5 independent shots are already all-floor by step 5.
INFERENCE_LATE_LO = 5
INFERENCE_LATE_DROP_MIN = 0.005
# Fig 5a steps 4–10 vs digitized SEA. n_obs=10 leftover-onset filling sits
# ~0.02 above paper here; n_obs=6 reaches the 150 ms floor at step 6.
INFERENCE_MID_LO = 4
INFERENCE_MID_SEA_MAE_MAX = 0.012
INFERENCE_PAPER_Y_TO_NORM = 1000.0
INFERENCE_EARLY_SERIES_50HZ = ("Baseline 50Hz", "SEA-DBS 50Hz")
INFERENCE_EARLY_SERIES_30HZ = ("Baseline 30Hz", "SEA-DBS 30Hz")
INFERENCE_30HZ_PEARSON_MIN = 0.70
INFERENCE_30HZ_EARLY_MAE_MAX = 0.025


def _paper_early_norm(stem: str, series: str, n: int = INFERENCE_EARLY_N) -> np.ndarray | None:
    path = curves_path(stem)
    if not path.is_file():
        return None
    curves = load_refined(path)
    if series not in curves:
        return None
    x, y = curves[series]
    yn = np.asarray(y, dtype=float) / INFERENCE_PAPER_Y_TO_NORM
    out: list[float] = []
    for step in range(n):
        val = window_mean(x, yn, lo=step - 0.35, hi=step + 0.35)
        if val is None or not np.isfinite(val):
            return None
        out.append(float(val))
    return np.asarray(out, dtype=float)


def ravivarapu_inference_gates(
    baseline_trace: Sequence[float],
    paper_trace: Sequence[float],
    *,
    carrier_hz: float,
    sea_trace_50hz: Sequence[float] | None = None,
    baseline_trace_50hz: Sequence[float] | None = None,
    n_expected: int = 11,
) -> dict[str, Any]:
    """Shape gates for Fig 5a/5b on **normalized** PSD (~0.3–0.5).

    Ordering: 11 samples, shared start, both decline, SEA below Baseline at
    the end, SEA steeper drop, correct carrier. 30 Hz also needs a weaker
    end than the 50 Hz panel. Fig 5a (50 Hz) also checks steps 0–5 against
    digitized paper with generous MAE / drop tols, steps 4–10 SEA MAE vs
    digitized paper, and steps 5–10 still declining (not a leftover-onset
    fill that sits above the 50 Hz floor).

    Paper crops label the same biomarker as raw ~300–480 (×1000 vs Fig 4a).
    Traces here are ``p_beta_norm``. ``n_expected`` is PSD samples: t=0
    untreated plus 10 stimulation steps (paper x-axis 0–10).
    """
    b = _as_fy(baseline_trace)
    p = _as_fy(paper_trace)
    b0, b_end = float(b[0]), float(b[-1])
    p0, p_end = float(p[0]), float(p[-1])
    b_drop = b0 - b_end
    p_drop = p0 - p_end
    b1 = float(b[1]) if b.size > 1 else b0
    p1 = float(p[1]) if p.size > 1 else p0
    b2 = float(b[2]) if b.size > 2 else b_end
    p2 = float(p[2]) if p.size > 2 else p_end
    gates: dict[str, bool] = {
        "n_steps_ok": int(b.size) == n_expected and int(p.size) == n_expected,
        "shared_start": abs(p0 - b0) < INFERENCE_SHARED_START_MAX,
        "baseline_declines": b_drop > INFERENCE_DECLINE_MIN,
        "paper_declines": p_drop > INFERENCE_DECLINE_MIN,
        "paper_end_below_baseline": p_end < b_end,
        "paper_steeper_drop": p_drop > b_drop,
        "carrier_hz_ok": carrier_hz in (30.0, 50.0),
    }
    if carrier_hz == 30.0 and sea_trace_50hz is not None and baseline_trace_50hz is not None:
        sea_30 = float(_as_fy(paper_trace)[-1])
        sea_50 = float(_as_fy(sea_trace_50hz)[-1])
        base_30 = float(_as_fy(baseline_trace)[-1])
        base_50 = float(_as_fy(baseline_trace_50hz)[-1])
        gates["weaker_than_50hz_sea"] = sea_30 > sea_50
        gates["weaker_than_50hz_baseline"] = base_30 > base_50
    early_mae_b = float("nan")
    early_mae_p = float("nan")
    early_mae_p_35 = float("nan")
    early_drop_p = float("nan")
    mid_mae_p = float("nan")
    if carrier_hz == 50.0 and b.size >= INFERENCE_EARLY_N and p.size >= INFERENCE_EARLY_N:
        paper_b = _paper_early_norm("fig5a", INFERENCE_EARLY_SERIES_50HZ[0])
        paper_s = _paper_early_norm("fig5a", INFERENCE_EARLY_SERIES_50HZ[1])
        paper_s_full = _paper_early_norm("fig5a", INFERENCE_EARLY_SERIES_50HZ[1], n=11)
        b_early = b[:INFERENCE_EARLY_N]
        p_early = p[:INFERENCE_EARLY_N]
        if paper_b is not None and paper_s is not None:
            early_mae_b = float(np.mean(np.abs(b_early - paper_b)))
            early_mae_p = float(np.mean(np.abs(p_early - paper_s)))
            early_mae_p_35 = float(np.mean(np.abs(p_early[3:] - paper_s[3:])))
            early_drop_p = float(p_early[0] - p_early[-1])
            early_drop_b = float(b_early[0] - b_early[-1])
            gates["shared_start_near_paper"] = bool(
                abs(b0 - float(paper_b[0])) <= 0.02 and abs(p0 - float(paper_s[0])) <= 0.02
            )
            gates["early_mae_baseline"] = early_mae_b <= INFERENCE_EARLY_MAE_MAX
            gates["early_mae_sea"] = early_mae_p <= INFERENCE_EARLY_MAE_MAX
            gates["early_mae_sea_3_5"] = early_mae_p_35 <= INFERENCE_EARLY_SEA_MAE_3_5_MAX
            gates["early_sea_declines"] = early_drop_p > INFERENCE_EARLY_SEA_DROP_MIN
            gates["early_baseline_declines"] = early_drop_b > INFERENCE_DECLINE_MIN
            # Paper: Baseline above SEA from step 1 through 5, not overlaid.
            gates["early_sea_below_baseline"] = bool(
                np.all(p_early[1:] < b_early[1:])
            )
        if paper_s_full is not None and p.size >= 11:
            mid_mae_p = float(np.mean(np.abs(p[INFERENCE_MID_LO:] - paper_s_full[INFERENCE_MID_LO:])))
            gates["mid_mae_sea"] = mid_mae_p <= INFERENCE_MID_SEA_MAE_MAX
        # Paper keeps falling after step 5; independent n_obs=5 floors there.
        late_drop_b = float(b[INFERENCE_LATE_LO] - b_end)
        late_drop_p = float(p[INFERENCE_LATE_LO] - p_end)
        gates["late_baseline_declines"] = late_drop_b > INFERENCE_LATE_DROP_MIN
        gates["late_sea_declines"] = late_drop_p > INFERENCE_LATE_DROP_MIN
    if carrier_hz == 30.0 and b.size >= INFERENCE_EARLY_N and p.size >= INFERENCE_EARLY_N:
        paper_b = _paper_early_norm("fig5b", INFERENCE_EARLY_SERIES_30HZ[0])
        paper_s = _paper_early_norm("fig5b", INFERENCE_EARLY_SERIES_30HZ[1])
        paper_b_full = _paper_early_norm("fig5b", INFERENCE_EARLY_SERIES_30HZ[0], n=11)
        paper_s_full = _paper_early_norm("fig5b", INFERENCE_EARLY_SERIES_30HZ[1], n=11)
        b_early = b[:INFERENCE_EARLY_N]
        p_early = p[:INFERENCE_EARLY_N]
        if paper_b is not None and paper_s is not None:
            early_mae_b = float(np.mean(np.abs(b_early - paper_b)))
            early_mae_p = float(np.mean(np.abs(p_early - paper_s)))
            gates["early_mae_baseline"] = early_mae_b <= INFERENCE_30HZ_EARLY_MAE_MAX
            gates["early_mae_sea"] = early_mae_p <= INFERENCE_30HZ_EARLY_MAE_MAX
            # Paper Fig 5b: Baseline rises at step 1 or stays elevated above onset
            gates["early_baseline_rises"] = b1 >= b0 - 0.002
            # Paper Fig 5b: SEA-DBS delayed drop / plateau on steps 0-2 (drop <= 0.015 step 1, <= 0.025 step 2)
            gates["early_sea_plateau"] = bool((p0 - p1 <= 0.015) and (p0 - p2 <= 0.025))
            # Paper: Baseline above SEA from step 1 through 5
            gates["early_sea_below_baseline"] = bool(
                np.all(p_early[1:] < b_early[1:])
            )
        # Late window declines
        late_drop_b = float(b[INFERENCE_LATE_LO] - b_end)
        late_drop_p = float(p[INFERENCE_LATE_LO] - p_end)
        gates["late_baseline_declines"] = late_drop_b > INFERENCE_LATE_DROP_MIN
        gates["late_sea_declines"] = late_drop_p > INFERENCE_LATE_DROP_MIN
        # Trajectory correlations vs digitized paper
        if paper_b_full is not None and b.size >= 11:
            r_b = float(np.corrcoef(b[:11], paper_b_full)[0, 1])
            gates["pearson_baseline_min"] = bool(np.isfinite(r_b) and r_b >= INFERENCE_30HZ_PEARSON_MIN)
        if paper_s_full is not None and p.size >= 11:
            r_s = float(np.corrcoef(p[:11], paper_s_full)[0, 1])
            gates["pearson_sea_min"] = bool(np.isfinite(r_s) and r_s >= INFERENCE_30HZ_PEARSON_MIN)
    metrics = {
        "carrier_hz": carrier_hz,
        "b_start": b0,
        "b_end": b_end,
        "p_start": p0,
        "p_end": p_end,
        "b_drop": b_drop,
        "p_drop": p_drop,
        "b_step1": b1,
        "p_step1": p1,
        "b_step2": b2,
        "p_step2": p2,
        "b_drop_0_2": b0 - b2,
        "p_drop_0_2": p0 - p2,
        "early_mae_baseline": early_mae_b,
        "early_mae_sea": early_mae_p,
        "early_mae_sea_3_5": early_mae_p_35,
        "early_drop_sea_0_5": early_drop_p,
        "mid_mae_sea": mid_mae_p,
        "late_drop_baseline_5_10": (
            float(b[INFERENCE_LATE_LO] - b_end) if b.size > INFERENCE_LATE_LO else float("nan")
        ),
        "late_drop_sea_5_10": (
            float(p[INFERENCE_LATE_LO] - p_end) if p.size > INFERENCE_LATE_LO else float("nan")
        ),
    }
    return _gate_pack(gates, metrics)


def ravivarapu_fig6_gates(
    traces: dict[str, Sequence[float]],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    required = (
        "Baseline",
        "Baseline + PTQ(fp16)",
        "SEA-DBS",
        "SEA-DBS + PTQ(fp16)",
    )
    present = all(k in traces for k in required)
    if not present:
        return _gate_pack(
            {"four_series_present": False},
            {"keys": list(traces.keys())},
            notes=["missing one or more Fig 6 series labels"],
        )

    b = _as_fy(traces["Baseline"])
    b_ptq = _as_fy(traces["Baseline + PTQ(fp16)"])
    sea = _as_fy(traces["SEA-DBS"])
    sea_ptq = _as_fy(traces["SEA-DBS + PTQ(fp16)"])

    b_late = float(np.mean(b[-max(1, len(b) // 2) :]))
    sea_late = float(np.mean(sea[-max(1, len(sea) // 2) :]))
    sea_ptq_late = float(np.mean(sea_ptq[-max(1, len(sea_ptq) // 2) :]))
    b_ptq_late = float(np.mean(b_ptq[-max(1, len(b_ptq) // 2) :]))
    sea_fp_late = float(np.mean(sea[-max(1, len(sea) // 2) :]))

    b_n = b if float(b[0]) < 10.0 else b / 1000.0
    b_ptq_n = b_ptq if float(b_ptq[0]) < 10.0 else b_ptq / 1000.0
    sea_n = sea if float(sea[0]) < 10.0 else sea / 1000.0
    sea_ptq_n = sea_ptq if float(sea_ptq[0]) < 10.0 else sea_ptq / 1000.0

    shared_start = (
        abs(float(b_n[0]) - float(sea_n[0])) < 0.05
        and abs(float(b_ptq_n[0]) - float(sea_ptq_n[0])) < 0.05
    )

    paper_b = _paper_early_norm("fig6", "Baseline")
    paper_s = _paper_early_norm("fig6", "SEA-DBS")

    early_mae_b = float("nan")
    early_mae_s = float("nan")
    shared_start_near_paper = True
    if paper_b is not None and paper_s is not None and b_n.size >= 6:
        early_mae_b = float(np.mean(np.abs(b_n[:6] - paper_b[:6])))
        early_mae_s = float(np.mean(np.abs(sea_n[:6] - paper_s[:6])))
        shared_start_near_paper = bool(
            abs(float(sea_n[0]) - float(paper_s[0])) <= 0.02
        )

    gates = {
        "four_series_present": True,
        "n_steps_ok": all(len(traces[k]) in (10, 11) for k in required),
        "shared_start": shared_start,
        "shared_start_near_paper": shared_start_near_paper,
        "early_mae_baseline": (
            early_mae_b <= 0.030 if np.isfinite(early_mae_b) else True
        ),
        "early_mae_sea": (
            early_mae_s <= 0.030 if np.isfinite(early_mae_s) else True
        ),
        "sea_below_baseline": sea_late < b_late,
        "sea_ptq_below_baseline": sea_ptq_late < b_late,
        "sea_ptq_tracks_fp32": rel_close(sea_ptq_late, sea_fp_late, tol=DEFAULT_REL_TOL + 0.15),
        "baseline_ptq_near_or_above_baseline": b_ptq_late >= b_late * 0.95,
        "ptq_traces_distinct": (
            not np.allclose(b, b_ptq, rtol=0.0, atol=1e-9)
            and not np.allclose(sea, sea_ptq, rtol=0.0, atol=1e-9)
        ),
    }
    metrics = {
        "b_late": b_late,
        "b_ptq_late": b_ptq_late,
        "sea_late": sea_late,
        "sea_ptq_late": sea_ptq_late,
        "early_mae_baseline": early_mae_b,
        "early_mae_sea": early_mae_s,
    }
    return _gate_pack(gates, metrics)


def ravivarapu_fig7_gates(
    traces: dict[str, Sequence[float]],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    required = ("baseline", "baseline-pm", "baseline-gs", "paper")
    if not all(k in traces for k in required):
        return _gate_pack(
            {"four_variants_present": False},
            {"keys": list(traces.keys())},
            notes=["missing ablation variant"],
        )

    tail = {k: float(np.mean(_as_fy(traces[k])[-max(1, len(traces[k]) // 2) :])) for k in required}
    gs = tail["baseline-gs"]
    pm = tail["baseline-pm"]
    base = tail["baseline"]
    sea = tail["paper"]

    sea_arr = _as_fy(traces["paper"])
    base_arr = _as_fy(traces["baseline"])
    sea_n = sea_arr if float(sea_arr[0]) < 10.0 else sea_arr / 1000.0
    base_n = base_arr if float(base_arr[0]) < 10.0 else base_arr / 1000.0

    paper_b = _paper_early_norm("fig7", "Baseline")
    paper_s = _paper_early_norm("fig7", "SEA-DBS")

    early_mae_b = float("nan")
    early_mae_s = float("nan")
    shared_start_near_paper = True
    if paper_b is not None and paper_s is not None and sea_n.size >= 6:
        early_mae_b = float(np.mean(np.abs(base_n[:6] - paper_b[:6])))
        early_mae_s = float(np.mean(np.abs(sea_n[:6] - paper_s[:6])))
        shared_start_near_paper = bool(
            abs(float(sea_n[0]) - float(paper_s[0])) <= 0.02
        )

    pm_arr = _as_fy(traces["baseline-pm"])
    pm_n = pm_arr if float(pm_arr[0]) < 10.0 else pm_arr / 1000.0
    baseline_above_pm_early = bool(np.all(base_n[:5] >= pm_n[:5] - 1e-6) and np.any(base_n[:5] > pm_n[:5] + 1e-4))

    gates = {
        "four_variants_present": True,
        "n_steps_ok": all(len(traces[k]) in (10, 11) for k in required),
        "shared_start": abs(float(base_n[0]) - float(sea_n[0])) < 0.05,
        "shared_start_near_paper": shared_start_near_paper,
        "early_mae_baseline": (
            early_mae_b <= 0.030 if np.isfinite(early_mae_b) else True
        ),
        "early_mae_sea": (
            early_mae_s <= 0.030 if np.isfinite(early_mae_s) else True
        ),
        "baseline_above_pm_early": baseline_above_pm_early,
        "sea_dbs_lowest_tail": sea <= min(tail.values()) + 1e-6,
        "gs_highest_or_near_highest_tail": gs >= max(tail.values()) - 5.0,
        "pm_not_sea": abs(pm - base) <= abs(pm - sea),
    }
    return _gate_pack(gates, {"tail_means": tail, "early_mae_baseline": early_mae_b, "early_mae_sea": early_mae_s})
