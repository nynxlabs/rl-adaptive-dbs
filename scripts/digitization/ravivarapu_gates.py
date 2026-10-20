"""Digitization-anchored gate helpers for Ravivarapu (SEA-DBS) panels.

Every threshold is anchored on the refined paper digitization
(``artifacts/figures/papers/ravivarapu/paper_digitization/curves_*.json``):
ordering, shared starts, window levels, drops, gaps and timing vs the paper,
with tolerances sized to the paper's own effect and our seed-to-seed noise.
A gate that needs digitization fails closed when it is missing. Panel notes:
``docs/controllers/sea_dbs/figures/<panel>.md``; gate audit (oct 14 2026):
``docs/controllers/sea_dbs/figures/4a.md`` § Gate audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from paper_gates import (
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
    # Shape tier (14 gates)
    "n_episodes_ok": "shape",
    "shared_start": "shape",
    "baseline_declines": "shape",
    "sea_below_baseline_mid": "shape",
    "sea_below_baseline_midlate": "shape",
    "sea_below_baseline_late": "shape",
    "sea_steeper_drop_than_baseline": "shape",
    "gap_widens_over_training": "shape",
    "drop_timing_baseline": "shape",
    "drop_timing_sea": "shape",
    "gradual_decline_baseline": "shape",
    "gradual_decline_sea": "shape",
    "pearson_baseline_min": "shape",
    "pearson_sea_min": "shape",
    # Full tier (14 gates)
    "shared_start_near_paper": "full",
    "baseline_drop_vs_paper": "full",
    "sea_drop_vs_paper": "full",
    "baseline_mid_near_paper": "full",
    "baseline_midlate_near_paper": "full",
    "baseline_late_near_paper": "full",
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
    "sea_faster_initial_rise": "shape",
    "rise_timing_baseline": "shape",
    "rise_timing_sea": "shape",
    "gradual_rise_baseline": "shape",
    "gradual_rise_sea": "shape",
    "pearson_baseline_min": "shape",
    "pearson_sea_min": "shape",
    # Full tier (6 gates)
    "eq7_consistent_with_4a": "full",
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


# Fig 4a start level: |ours − paper| on the ep 0–15 mean. 0.02 is the same
# start tolerance Figs 5–7 use, and in reward units it matches Fig 4b's 30 %
# (Eq. (7) at 0.45 ± 0.02 spans −0.64 … −1.44 around the paper's −0.9).
FIG4A_START_ABS_TOL = 0.02
# Late/early ratio vs paper (Baseline 0.833, SEA-DBS 0.762). With our raw
# per-episode noise (sd ~0.019) a perfect replication's ratio wanders ~2 %;
# 0.30 let a flat Baseline (ratio 1.0) pass.
FIG4A_RATIO_TOL = 0.08
# Window level vs paper, both variants (Baseline levels were never checked).
FIG4A_LEVEL_REL_TOL = 0.12
# Fig 4b ordering / gap gates average over ep 90–150. Our per-episode reward
# noise (sd ~0.19) against the paper's 0.05 late gap made the ep 120–150
# versions coin-flips for a perfect replication (60–85 % pass); over 90–150
# they pass 95–98 % while a Baseline/SEA tie still passes only ~50 % (ordering)
# or ~20 % (gap).
FIG4B_GAP_LATE_LO = 90.0
FIG4B_LATE_GAP_FRAC = 0.35
# Fig 4b vs Eq. (7) of Fig 4a, episode by episode: the windowed mean of
# (reward − Eq. (7)(PSD)) is ≤ 0.038 on the digitized paper curves (Pearson r
# 0.996 / 0.987); allow 0.05. Per-episode, so between-episode noise cancels.
FIG4B_EQ7_WINDOWS: tuple[tuple[float, float], ...] = (
    (0.0, 15.0),
    (40.0, 80.0),
    (70.0, 110.0),
    (120.0, 150.0),
)
FIG4B_EQ7_MAX_ABS = 0.05
EQ7_BETA_THRESHOLD = 0.35
EQ7_SCALE = 10.0


def eq7_reward(p_beta: Any) -> np.ndarray:
    """Paper Eq. (7) on normalized PSD (β_t = 0.35, (10 Δ)² with sign)."""
    p = np.asarray(p_beta, dtype=float)
    d = (p - EQ7_BETA_THRESHOLD) * EQ7_SCALE
    return np.where(p < EQ7_BETA_THRESHOLD, d * d, -(d * d))


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
    pearson_baseline_min: float = 0.58,
    pearson_sea_min: float = 0.58,
    n_expected: int = 150,
    shared_start_tol: float = 0.05,
) -> dict[str, Any]:
    """Unified Fig 4a episode PSD gates vs digitized paper learning curves.

    Shape tier: 150 episodes, shared untreated start, a genuine Baseline drop,
    SEA steeper and lower than Baseline from mid training on, widening gap,
    not front-loaded drop by ep 50, gradual mid-to-late decline, and Pearson r.
    Full tier: levels vs paper (start within 0.02; mid / ep 70–110 / late
    windows for both variants), drops vs paper, bounded gaps, late/early ratios.
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
        "shared_start_near_paper": bool(
            np.isfinite(b_early)
            and np.isfinite(s_early)
            and abs(b_early - pb_early) <= FIG4A_START_ABS_TOL
            and abs(s_early - ps_early) <= FIG4A_START_ABS_TOL
        ),
        "baseline_drop_vs_paper": bool(
            np.isfinite(b_drop) and np.isfinite(pb_drop) and b_drop >= drop_frac_of_paper * pb_drop
        ),
        "sea_drop_vs_paper": bool(
            np.isfinite(s_drop) and np.isfinite(ps_drop) and s_drop >= drop_frac_of_paper * ps_drop
        ),
        "baseline_mid_near_paper": rel_close(b_mid, pb_mid, tol=FIG4A_LEVEL_REL_TOL),
        "baseline_midlate_near_paper": rel_close(
            b_midlate_w, pb_midlate_w, tol=FIG4A_LEVEL_REL_TOL
        ),
        "baseline_late_near_paper": rel_close(b_late, pb_late, tol=FIG4A_LEVEL_REL_TOL),
        "sea_mid_near_paper": rel_close(s_mid, ps_mid, tol=FIG4A_LEVEL_REL_TOL),
        "sea_midlate_near_paper": rel_close(
            s_midlate_w, ps_midlate_w, tol=FIG4A_LEVEL_REL_TOL
        ),
        "sea_late_near_paper": rel_close(s_late, ps_late, tol=FIG4A_LEVEL_REL_TOL),
        "mid_gap_near_paper": bool(
            np.isfinite(gap_mid)
            and np.isfinite(p_gap_mid)
            and p_gap_mid > 0
            and 0.40 * p_gap_mid <= gap_mid <= 1.80 * p_gap_mid
        ),
        "midlate_gap_near_paper": bool(
            np.isfinite(gap_midlate)
            and np.isfinite(p_gap_midlate)
            and p_gap_midlate > 0
            and 0.40 * p_gap_midlate <= gap_midlate <= 1.75 * p_gap_midlate
        ),
        "late_gap_near_paper": bool(
            np.isfinite(gap)
            and np.isfinite(p_gap)
            and p_gap > 0
            and 0.50 * p_gap <= gap <= 1.60 * p_gap
        ),
        "late_early_ratio_baseline_near_paper": ratio_close(
            b_late, b_early, pb_late, pb_early, tol=FIG4A_RATIO_TOL
        ),
        "late_early_ratio_sea_near_paper": ratio_close(
            s_late, s_early, ps_late, ps_early, tol=FIG4A_RATIO_TOL
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
    baseline_psd: Sequence[float] | None = None,
    sea_psd: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Unified Fig 4b episode reward gates vs digitized paper learning curves.

    Shape tier: 150 episodes, shared untreated start, genuine learning rise
    on both variants, SEA steeper and higher than Baseline late (ep 90–150),
    SEA ahead mid-training and faster to its plateau, not front-loaded rise by
    ep 50, gradual mid-to-late increase, and Pearson r.
    Full tier: Fig 4b equals Eq. (7) of the paired Fig 4a PSD window by window
    (``baseline_psd`` / ``sea_psd``; fails closed without them), start and
    rises vs paper, late and ep 70–110 gaps vs paper.
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

    # Ordering / gap gates average ep 90–150 (FIG4B_GAP_LATE_LO).
    b_late_gap_w = window_mean(x, b, lo=FIG4B_GAP_LATE_LO)
    s_late_gap_w = window_mean(x, s, lo=FIG4B_GAP_LATE_LO)
    pb_late_gap_w = window_mean(pbx, pby, lo=FIG4B_GAP_LATE_LO)
    ps_late_gap_w = window_mean(psx, psy, lo=FIG4B_GAP_LATE_LO)
    gap = s_late_gap_w - b_late_gap_w
    gap_midlate = s_midlate_w - b_midlate_w
    p_gap = ps_late_gap_w - pb_late_gap_w
    p_gap_midlate = ps_midlate_w - pb_midlate_w
    b_steep = b_late_gap_w - b_early
    s_steep = s_late_gap_w - s_early

    # Paper: SEA-DBS has "a faster initial learning rate" — it covers more of
    # its total rise by the ep 40–80 window (paper 0.79 vs Baseline 0.66).
    def _mid_share(early: float, mid: float, late: float) -> float:
        total = late - early
        if not np.isfinite(total) or total <= 1e-9:
            return float("nan")
        return float((mid - early) / total)

    b_mid_share = _mid_share(b_early, b_mid, b_late_gap_w)
    s_mid_share = _mid_share(s_early, s_mid, s_late_gap_w)
    pb_mid_share = _mid_share(pb_early, pb_mid, pb_late_gap_w)
    ps_mid_share = _mid_share(ps_early, ps_mid, ps_late_gap_w)

    # Fig 4b vs Eq. (7) of Fig 4a, window by window.
    eq7_err_b = float("nan")
    eq7_err_s = float("nan")
    if baseline_psd is not None and sea_psd is not None:
        n_psd = min(n, len(baseline_psd), len(sea_psd))
        if n_psd >= n:
            resid_b = b - eq7_reward(_as_fy(baseline_psd[:n]))
            resid_s = s - eq7_reward(_as_fy(sea_psd[:n]))
            eq7_err_b = max(abs(window_mean(x, resid_b, lo=lo, hi=hi)) for lo, hi in FIG4B_EQ7_WINDOWS)
            eq7_err_s = max(abs(window_mean(x, resid_s, lo=lo, hi=hi)) for lo, hi in FIG4B_EQ7_WINDOWS)

    shared_start_dist = abs(b_early - s_early) / (abs(b_early) + 1e-6)

    gates = {
        # Shape tier
        "n_episodes_ok": n >= n_expected,
        "shared_start": bool(
            np.isfinite(b_early)
            and np.isfinite(s_early)
            and shared_start_dist <= 0.25
        ),
        "baseline_rises": bool(np.isfinite(b_rise) and b_rise >= 0.25 * pb_rise),
        "sea_rises": bool(np.isfinite(s_rise) and s_rise >= 0.25 * ps_rise),
        "sea_above_baseline_late": bool(
            np.isfinite(s_late_gap_w) and np.isfinite(b_late_gap_w) and s_late_gap_w > b_late_gap_w
        ),
        "sea_steeper_rise_than_baseline": bool(
            np.isfinite(s_steep) and np.isfinite(b_steep) and s_steep > b_steep
        ),
        "paper_pull_ahead_mid": bool(
            np.isfinite(s_mid) and np.isfinite(b_mid) and s_mid > b_mid
        ),
        "sea_faster_initial_rise": bool(
            np.isfinite(s_mid_share) and np.isfinite(b_mid_share) and s_mid_share > b_mid_share
        ),
        "rise_timing_baseline": _drop_timing_ok(b_rise_frac, pb_rise_frac),
        "rise_timing_sea": _drop_timing_ok(s_rise_frac, ps_rise_frac),
        "gradual_rise_baseline": bool(
            np.isfinite(b_midlate)
            and np.isfinite(pb_midlate)
            and b_midlate >= profile_frac_of_paper * pb_midlate
        ),
        "gradual_rise_sea": bool(
            np.isfinite(s_midlate)
            and np.isfinite(ps_midlate)
            and s_midlate >= profile_frac_of_paper * ps_midlate
        ),
        "pearson_baseline_min": bool(
            np.isfinite(shape_b) and shape_b >= pearson_baseline_min
        ),
        "pearson_sea_min": bool(
            np.isfinite(shape_s) and shape_s >= pearson_sea_min
        ),
        # Full tier
        "eq7_consistent_with_4a": bool(
            np.isfinite(eq7_err_b)
            and np.isfinite(eq7_err_s)
            and eq7_err_b <= FIG4B_EQ7_MAX_ABS
            and eq7_err_s <= FIG4B_EQ7_MAX_ABS
        ),
        "shared_start_near_paper": rel_close(s_early, ps_early, tol=0.30)
        and rel_close(b_early, pb_early, tol=0.30),
        "baseline_rise_vs_paper": bool(
            np.isfinite(b_rise)
            and np.isfinite(pb_rise)
            and b_rise >= rise_frac_of_paper * pb_rise
        ),
        "sea_rise_vs_paper": bool(
            np.isfinite(s_rise)
            and np.isfinite(ps_rise)
            and s_rise >= rise_frac_of_paper * ps_rise
        ),
        "late_gap_near_paper": bool(
            np.isfinite(gap)
            and np.isfinite(p_gap)
            and gap >= FIG4B_LATE_GAP_FRAC * p_gap
        ),
        "midlate_gap_near_paper": bool(
            np.isfinite(gap_midlate)
            and np.isfinite(p_gap_midlate)
            and gap_midlate >= 0.40 * p_gap_midlate
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
        "late_gap_90_150": gap,
        "paper_late_gap_90_150": p_gap,
        "b_late_90_150": b_late_gap_w,
        "s_late_90_150": s_late_gap_w,
        "b_mid_share": b_mid_share,
        "s_mid_share": s_mid_share,
        "paper_b_mid_share": pb_mid_share,
        "paper_s_mid_share": ps_mid_share,
        "eq7_max_abs_err_baseline": eq7_err_b,
        "eq7_max_abs_err_sea": eq7_err_s,
        "midlate_gap": gap_midlate,
        "paper_midlate_gap": p_gap_midlate,
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


# Figs 5–7 inference gates on normalized PSD (paper crop PSD / 1000).
#
# Every magnitude is anchored on the digitized paper curve of the same series:
# MAE caps are a share of that series' step 0→10 drop (the effect the panel
# shows), level and start bands are ±0.02 (the Fig 4a / 5–7 start tolerance).
# Gates that need digitization fail closed when it is missing.
INFERENCE_DECLINE_MIN = 0.0
INFERENCE_SHARED_START_MAX = 0.05
INFERENCE_START_ABS_TOL = 0.02
INFERENCE_LEVEL_ABS_TOL = 0.02
INFERENCE_EARLY_N = 6
INFERENCE_N_STEPS = 11
# Steps 0–5 MAE ≤ 20 % of the paper series' 0→10 drop (50 Hz Baseline 0.021,
# SEA-DBS 0.030; 30 Hz 0.012 / 0.014).
INFERENCE_EARLY_MAE_DROP_FRAC = 0.20
# Narrow SEA-DBS windows (steps 3–5, 4–10) ≤ 10 % of the paper SEA drop (0.015).
INFERENCE_SEA_WINDOW_MAE_DROP_FRAC = 0.10
# SEA-DBS falls at least 40 % of the paper's steps 0→5 drop by step 5 (0.048).
INFERENCE_EARLY_SEA_DROP_FRAC = 0.40
# Steps 5–10: paper keeps falling (SEA ~0.338→0.310, Baseline ~0.404→0.356).
INFERENCE_LATE_LO = 5
INFERENCE_LATE_DROP_MIN = 0.005
INFERENCE_MID_LO = 4
# Step 0→10 drop within [0.5, 1.5] × the paper's (50 Hz B 0.104 / S 0.148;
# 30 Hz B 0.062 / S 0.072).
INFERENCE_DROP_BAND = (0.5, 1.5)
# Tail (steps 6–10) Baseline − SEA-DBS gap within [0.5, 1.6] × paper.
INFERENCE_TAIL_GAP_BAND = (0.5, 1.6)
INFERENCE_PAPER_Y_TO_NORM = 1000.0
INFERENCE_EARLY_SERIES_50HZ = ("Baseline 50Hz", "SEA-DBS 50Hz")
INFERENCE_EARLY_SERIES_30HZ = ("Baseline 30Hz", "SEA-DBS 30Hz")
INFERENCE_30HZ_PEARSON_MIN = 0.70
# Fig 5b steps 1–5: the digitized SEA-DBS sits up to 0.0024 (window mean) /
# 0.0032 (linear interpolation) above Baseline at step 4.
INFERENCE_30HZ_CROSS_TOL = 0.006
# Figs 5a, 6 (fp32) and 7 plot one rollout: the digitized Baseline and SEA-DBS
# curves differ across the three panels by ≤ 0.0024.
SHARED_ROLLOUT_MAX_ABS = 0.005
# Fig 6: PTQ vs fp32 of the same variant. Paper tails differ by 0.012 / 0.009
# and pointwise by ≤ 0.032 (SEA-DBS step 7).
PTQ_TAIL_MAX_ABS = 0.03
PTQ_POINT_MAX_ABS = 0.04


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


def _norm(y: Sequence[float]) -> np.ndarray:
    arr = _as_fy(y)
    return arr if arr.size == 0 or float(arr[0]) < 10.0 else arr / INFERENCE_PAPER_Y_TO_NORM


def _tail(y: np.ndarray) -> float:
    """Mean of the last half of the trace (steps 6–10 of 11 samples)."""
    if y.size == 0:
        return float("nan")
    return float(np.mean(y[-max(1, y.size // 2) :]))


def _mae(ours: np.ndarray, paper: np.ndarray | None, lo: int = 0, hi: int | None = None) -> float:
    if paper is None:
        return float("nan")
    hi = min(ours.size, paper.size) if hi is None else hi
    if ours.size < hi or paper.size < hi or hi <= lo:
        return float("nan")
    return float(np.mean(np.abs(ours[lo:hi] - paper[lo:hi])))


def _le(value: float, cap: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(cap) and value <= cap)


def _in_band(value: float, ref: float, band: tuple[float, float]) -> bool:
    return bool(
        np.isfinite(value) and np.isfinite(ref) and ref > 0 and band[0] * ref <= value <= band[1] * ref
    )


def _paper_drop(paper: np.ndarray | None) -> float:
    if paper is None or paper.size < 2:
        return float("nan")
    return float(paper[0] - paper[-1])


def _shared_rollout_gate(
    ours: dict[str, np.ndarray],
    fig5a: dict[str, Sequence[float]] | None,
    pairs: Sequence[tuple[str, str]],
) -> tuple[bool, float]:
    """Our fp32 Baseline / SEA-DBS traces equal the Fig 5a rollout (paper: same rollout)."""
    if not fig5a:
        return False, float("nan")
    worst = 0.0
    for ours_key, fig5a_key in pairs:
        if fig5a_key not in fig5a or ours_key not in ours:
            return False, float("nan")
        a = ours[ours_key]
        b = _norm(fig5a[fig5a_key])
        if a.size != b.size or a.size == 0:
            return False, float("nan")
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst <= SHARED_ROLLOUT_MAX_ABS, worst


def ravivarapu_inference_gates(
    baseline_trace: Sequence[float],
    paper_trace: Sequence[float],
    *,
    carrier_hz: float,
    sea_trace_50hz: Sequence[float] | None = None,
    baseline_trace_50hz: Sequence[float] | None = None,
    n_expected: int = INFERENCE_N_STEPS,
) -> dict[str, Any]:
    """Fig 5a / 5b gates on **normalized** PSD (~0.3–0.5).

    Ordering: 11 samples, shared start, both decline, SEA below Baseline at
    the end, SEA steeper drop, correct carrier. Magnitudes vs the digitized
    paper series: start, step 0→10 drops, tail gap, steps 0–5 MAE, and (50 Hz)
    SEA-DBS steps 3–5 / 4–10 MAE and continued decline over steps 5–10.
    Fig 5b also needs both 30 Hz drops positive but smaller than the paired
    50 Hz drops (``*_trace_50hz``; fails closed without them).

    Paper crops label the same biomarker as raw ~300–480 (×1000 vs Fig 4a).
    Traces here are ``p_beta_norm``. ``n_expected`` is PSD samples: t=0
    untreated plus 10 stimulation steps (paper x-axis 0–10).
    """
    b = _norm(baseline_trace)
    p = _norm(paper_trace)
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
    stem = "fig5a" if carrier_hz == 50.0 else "fig5b"
    series = INFERENCE_EARLY_SERIES_50HZ if carrier_hz == 50.0 else INFERENCE_EARLY_SERIES_30HZ
    paper_b = _paper_early_norm(stem, series[0], n=INFERENCE_N_STEPS)
    paper_s = _paper_early_norm(stem, series[1], n=INFERENCE_N_STEPS)
    pb_drop = _paper_drop(paper_b)
    ps_drop = _paper_drop(paper_s)
    full = b.size >= INFERENCE_N_STEPS and p.size >= INFERENCE_N_STEPS
    have_paper = paper_b is not None and paper_s is not None and full

    early_mae_b = _mae(b, paper_b, 0, INFERENCE_EARLY_N) if have_paper else float("nan")
    early_mae_p = _mae(p, paper_s, 0, INFERENCE_EARLY_N) if have_paper else float("nan")
    b_tail, p_tail = _tail(b), _tail(p)
    pb_tail = _tail(paper_b) if paper_b is not None else float("nan")
    ps_tail = _tail(paper_s) if paper_s is not None else float("nan")
    gates["shared_start_near_paper"] = bool(
        have_paper
        and abs(b0 - float(paper_b[0])) <= INFERENCE_START_ABS_TOL
        and abs(p0 - float(paper_s[0])) <= INFERENCE_START_ABS_TOL
    )
    gates["early_mae_baseline"] = _le(early_mae_b, INFERENCE_EARLY_MAE_DROP_FRAC * pb_drop)
    gates["early_mae_sea"] = _le(early_mae_p, INFERENCE_EARLY_MAE_DROP_FRAC * ps_drop)
    gates["baseline_drop_near_paper"] = have_paper and _in_band(b_drop, pb_drop, INFERENCE_DROP_BAND)
    gates["sea_drop_near_paper"] = have_paper and _in_band(p_drop, ps_drop, INFERENCE_DROP_BAND)
    gates["tail_gap_near_paper"] = have_paper and _in_band(
        b_tail - p_tail, pb_tail - ps_tail, INFERENCE_TAIL_GAP_BAND
    )

    early_mae_p_35 = float("nan")
    early_drop_p = float("nan")
    mid_mae_p = float("nan")
    r_b = float("nan")
    r_s = float("nan")
    if carrier_hz == 50.0:
        early_mae_p_35 = _mae(p, paper_s, 3, INFERENCE_EARLY_N) if have_paper else float("nan")
        mid_mae_p = _mae(p, paper_s, INFERENCE_MID_LO, INFERENCE_N_STEPS) if have_paper else float("nan")
        early_drop_p = float(p[0] - p[INFERENCE_EARLY_N - 1]) if p.size >= INFERENCE_EARLY_N else float("nan")
        early_drop_b = float(b[0] - b[INFERENCE_EARLY_N - 1]) if b.size >= INFERENCE_EARLY_N else float("nan")
        ps_early_drop = (
            float(paper_s[0] - paper_s[INFERENCE_EARLY_N - 1]) if paper_s is not None else float("nan")
        )
        gates["early_mae_sea_3_5"] = _le(early_mae_p_35, INFERENCE_SEA_WINDOW_MAE_DROP_FRAC * ps_drop)
        gates["early_sea_declines"] = bool(
            np.isfinite(early_drop_p)
            and np.isfinite(ps_early_drop)
            and early_drop_p > INFERENCE_EARLY_SEA_DROP_FRAC * ps_early_drop
        )
        gates["early_baseline_declines"] = bool(np.isfinite(early_drop_b) and early_drop_b > INFERENCE_DECLINE_MIN)
        # Paper: Baseline above SEA from step 1 through 5, not overlaid.
        gates["early_sea_below_baseline"] = bool(
            full and np.all(p[1:INFERENCE_EARLY_N] < b[1:INFERENCE_EARLY_N])
        )
        gates["mid_mae_sea"] = _le(mid_mae_p, INFERENCE_SEA_WINDOW_MAE_DROP_FRAC * ps_drop)
    if carrier_hz == 30.0:
        # Paper Fig 5b: Baseline rises at step 1 or stays elevated above onset.
        gates["early_baseline_rises"] = b1 >= b0 - 0.002
        # Paper Fig 5b: SEA-DBS delayed drop / plateau on steps 0–2.
        gates["early_sea_plateau"] = bool((p0 - p1 <= 0.015) and (p0 - p2 <= 0.025))
        gates["early_sea_below_baseline"] = bool(
            full
            and np.all(p[1:INFERENCE_EARLY_N] <= b[1:INFERENCE_EARLY_N] + INFERENCE_30HZ_CROSS_TOL)
            and float(np.mean(p[1:INFERENCE_EARLY_N])) < float(np.mean(b[1:INFERENCE_EARLY_N]))
        )
        if have_paper:
            r_b = float(np.corrcoef(b[:INFERENCE_N_STEPS], paper_b)[0, 1])
            r_s = float(np.corrcoef(p[:INFERENCE_N_STEPS], paper_s)[0, 1])
        gates["pearson_baseline_min"] = bool(np.isfinite(r_b) and r_b >= INFERENCE_30HZ_PEARSON_MIN)
        gates["pearson_sea_min"] = bool(np.isfinite(r_s) and r_s >= INFERENCE_30HZ_PEARSON_MIN)
        # Paper: 30 Hz still suppresses, just less than 50 Hz (drops 0.062 /
        # 0.072 vs 0.104 / 0.148). Compare drops, not end levels — a rising
        # 30 Hz trace must not count as "weaker".
        b50 = _norm(baseline_trace_50hz) if baseline_trace_50hz is not None else None
        s50 = _norm(sea_trace_50hz) if sea_trace_50hz is not None else None
        b50_drop = float(b50[0] - b50[-1]) if b50 is not None and b50.size else float("nan")
        s50_drop = float(s50[0] - s50[-1]) if s50 is not None and s50.size else float("nan")
        gates["weaker_than_50hz_baseline"] = bool(np.isfinite(b50_drop) and 0.0 < b_drop < b50_drop)
        gates["weaker_than_50hz_sea"] = bool(np.isfinite(s50_drop) and 0.0 < p_drop < s50_drop)
    if full:
        late_drop_b = float(b[INFERENCE_LATE_LO] - b_end)
        late_drop_p = float(p[INFERENCE_LATE_LO] - p_end)
    else:
        late_drop_b = late_drop_p = float("nan")
    gates["late_baseline_declines"] = bool(np.isfinite(late_drop_b) and late_drop_b > INFERENCE_LATE_DROP_MIN)
    gates["late_sea_declines"] = bool(np.isfinite(late_drop_p) and late_drop_p > INFERENCE_LATE_DROP_MIN)
    metrics = {
        "carrier_hz": carrier_hz,
        "b_start": b0,
        "b_end": b_end,
        "p_start": p0,
        "p_end": p_end,
        "b_drop": b_drop,
        "p_drop": p_drop,
        "paper_b_drop": pb_drop,
        "paper_s_drop": ps_drop,
        "b_tail": b_tail,
        "p_tail": p_tail,
        "tail_gap": b_tail - p_tail,
        "paper_tail_gap": pb_tail - ps_tail,
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
        "pearson_baseline": r_b,
        "pearson_sea": r_s,
        "late_drop_baseline_5_10": late_drop_b,
        "late_drop_sea_5_10": late_drop_p,
    }
    notes = [] if have_paper else [f"paper digitization for {stem} missing — paper-anchored gates fail closed"]
    return _gate_pack(gates, metrics, notes=notes)


FIG6_SERIES = (
    "Baseline",
    "Baseline + PTQ(fp16)",
    "SEA-DBS",
    "SEA-DBS + PTQ(fp16)",
)


def ravivarapu_fig6_gates(
    traces: dict[str, Sequence[float]],
    *,
    fig5a_traces: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Fig 6 (FP16 PTQ @ 50 Hz) gates on normalized PSD.

    Paper claim: quantized SEA-DBS tracks full precision and still beats the
    Baseline. Gates: SEA-DBS below Baseline (fp32 and PTQ), each PTQ trace
    within a two-sided band of its fp32 trace, all four tails and the start vs
    paper, steps 0–5 MAE, and fp32 traces equal to the Fig 5a rollout
    (``fig5a_traces``; the paper plots the same rollout).
    """
    if not all(k in traces for k in FIG6_SERIES):
        return _gate_pack(
            {"four_series_present": False},
            {"keys": list(traces.keys())},
            notes=["missing one or more Fig 6 series labels"],
        )
    ours = {k: _norm(traces[k]) for k in FIG6_SERIES}
    b, b_ptq = ours["Baseline"], ours["Baseline + PTQ(fp16)"]
    sea, sea_ptq = ours["SEA-DBS"], ours["SEA-DBS + PTQ(fp16)"]
    tails = {k: _tail(v) for k, v in ours.items()}
    paper = {k: _paper_early_norm("fig6", k, n=INFERENCE_N_STEPS) for k in FIG6_SERIES}
    have_paper = all(v is not None for v in paper.values())
    paper_tails = {k: (_tail(v) if v is not None else float("nan")) for k, v in paper.items()}

    def _ptq_dev(fp: np.ndarray, q: np.ndarray) -> tuple[float, float]:
        if fp.size != q.size or fp.size == 0:
            return float("nan"), float("nan")
        return abs(_tail(q) - _tail(fp)), float(np.max(np.abs(q - fp)))

    b_tail_dev, b_point_dev = _ptq_dev(b, b_ptq)
    s_tail_dev, s_point_dev = _ptq_dev(sea, sea_ptq)
    early_mae_b = _mae(b, paper["Baseline"], 0, INFERENCE_EARLY_N)
    early_mae_s = _mae(sea, paper["SEA-DBS"], 0, INFERENCE_EARLY_N)
    shared_rollout, shared_dev = _shared_rollout_gate(
        {"Baseline": b, "SEA-DBS": sea}, fig5a_traces, (("Baseline", "baseline"), ("SEA-DBS", "paper"))
    )

    def _tail_near(key: str) -> bool:
        return bool(
            np.isfinite(tails[key])
            and np.isfinite(paper_tails[key])
            and abs(tails[key] - paper_tails[key]) <= INFERENCE_LEVEL_ABS_TOL
        )

    gates = {
        "four_series_present": True,
        "n_steps_ok": all(v.size == INFERENCE_N_STEPS for v in ours.values()),
        "shared_start": abs(float(b[0]) - float(sea[0])) < INFERENCE_SHARED_START_MAX
        and abs(float(b_ptq[0]) - float(sea_ptq[0])) < INFERENCE_SHARED_START_MAX,
        "shared_start_near_paper": bool(
            have_paper
            and abs(float(b[0]) - float(paper["Baseline"][0])) <= INFERENCE_START_ABS_TOL
            and abs(float(sea[0]) - float(paper["SEA-DBS"][0])) <= INFERENCE_START_ABS_TOL
        ),
        "early_mae_baseline": _le(early_mae_b, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["Baseline"])),
        "early_mae_sea": _le(early_mae_s, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["SEA-DBS"])),
        "fp32_matches_fig5a": shared_rollout,
        "sea_below_baseline": tails["SEA-DBS"] < tails["Baseline"],
        "sea_ptq_below_baseline": tails["SEA-DBS + PTQ(fp16)"] < tails["Baseline"],
        "sea_ptq_tracks_fp32": _le(s_tail_dev, PTQ_TAIL_MAX_ABS) and _le(s_point_dev, PTQ_POINT_MAX_ABS),
        "baseline_ptq_tracks_fp32": _le(b_tail_dev, PTQ_TAIL_MAX_ABS) and _le(b_point_dev, PTQ_POINT_MAX_ABS),
        "baseline_tail_near_paper": _tail_near("Baseline"),
        "sea_tail_near_paper": _tail_near("SEA-DBS"),
        "baseline_ptq_tail_near_paper": _tail_near("Baseline + PTQ(fp16)"),
        "sea_ptq_tail_near_paper": _tail_near("SEA-DBS + PTQ(fp16)"),
    }
    metrics = {
        "b_late": tails["Baseline"],
        "b_ptq_late": tails["Baseline + PTQ(fp16)"],
        "sea_late": tails["SEA-DBS"],
        "sea_ptq_late": tails["SEA-DBS + PTQ(fp16)"],
        "paper_tails": paper_tails,
        "baseline_ptq_tail_dev": b_tail_dev,
        "baseline_ptq_point_dev": b_point_dev,
        "sea_ptq_tail_dev": s_tail_dev,
        "sea_ptq_point_dev": s_point_dev,
        "fp32_vs_fig5a_max_abs": shared_dev,
        "early_mae_baseline": early_mae_b,
        "early_mae_sea": early_mae_s,
    }
    notes = [] if have_paper else ["fig6 digitization missing — paper-anchored gates fail closed"]
    return _gate_pack(gates, metrics, notes=notes)


# Fig 7 tail ordering must be a real separation, not a tie: paper gaps are
# ~0.04–0.05 (normalized PSD); require at least this much.
FIG7_MIN_SEPARATION = 0.005
FIG7_PAPER_SERIES = {
    "baseline": "Baseline",
    "baseline-pm": "Baseline + PM",
    "baseline-gs": "Baseline + GS",
    "paper": "SEA-DBS",
}


def ravivarapu_fig7_gates(
    traces: dict[str, Sequence[float]],
    *,
    fig5a_traces: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Fig 7 ablation gates on normalized PSD.

    Ordering of the tails (steps 6–10; paper GS 0.419 > Baseline 0.380 ≈
    PM 0.373 > SEA-DBS 0.323), each tail within ±0.02 of the paper, steps
    0–5 MAE for Baseline / SEA-DBS and full-trace MAE for +PM / +GS, the
    early Baseline-above-PM split, and Baseline / SEA-DBS equal to the Fig 5a
    rollout (``fig5a_traces``).
    """
    required = tuple(FIG7_PAPER_SERIES)
    if not all(k in traces for k in required):
        return _gate_pack(
            {"four_variants_present": False},
            {"keys": list(traces.keys())},
            notes=["missing ablation variant"],
        )
    ours = {k: _norm(traces[k]) for k in required}
    tail = {k: _tail(v) for k, v in ours.items()}
    gs, pm, base, sea = tail["baseline-gs"], tail["baseline-pm"], tail["baseline"], tail["paper"]
    paper = {k: _paper_early_norm("fig7", s, n=INFERENCE_N_STEPS) for k, s in FIG7_PAPER_SERIES.items()}
    have_paper = all(v is not None for v in paper.values())
    paper_tail = {k: (_tail(v) if v is not None else float("nan")) for k, v in paper.items()}

    early_mae_b = _mae(ours["baseline"], paper["baseline"], 0, INFERENCE_EARLY_N)
    early_mae_s = _mae(ours["paper"], paper["paper"], 0, INFERENCE_EARLY_N)
    mae_pm = _mae(ours["baseline-pm"], paper["baseline-pm"])
    mae_gs = _mae(ours["baseline-gs"], paper["baseline-gs"])
    base_n, pm_n, sea_n = ours["baseline"], ours["baseline-pm"], ours["paper"]
    baseline_above_pm_early = bool(
        base_n.size >= 5
        and pm_n.size >= 5
        and np.all(base_n[:5] >= pm_n[:5] - 1e-6)
        and np.any(base_n[:5] > pm_n[:5] + 1e-4)
    )
    shared_rollout, shared_dev = _shared_rollout_gate(
        ours, fig5a_traces, (("baseline", "baseline"), ("paper", "paper"))
    )

    def _tail_near(key: str) -> bool:
        return bool(
            np.isfinite(tail[key])
            and np.isfinite(paper_tail[key])
            and abs(tail[key] - paper_tail[key]) <= INFERENCE_LEVEL_ABS_TOL
        )

    gates = {
        "four_variants_present": True,
        "n_steps_ok": all(v.size == INFERENCE_N_STEPS for v in ours.values()),
        "shared_start": abs(float(base_n[0]) - float(sea_n[0])) < INFERENCE_SHARED_START_MAX,
        "shared_start_near_paper": bool(
            have_paper
            and abs(float(base_n[0]) - float(paper["baseline"][0])) <= INFERENCE_START_ABS_TOL
            and abs(float(sea_n[0]) - float(paper["paper"][0])) <= INFERENCE_START_ABS_TOL
        ),
        "early_mae_baseline": _le(early_mae_b, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["baseline"])),
        "early_mae_sea": _le(early_mae_s, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["paper"])),
        "mae_pm": _le(mae_pm, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["baseline-pm"])),
        "mae_gs": _le(mae_gs, INFERENCE_EARLY_MAE_DROP_FRAC * _paper_drop(paper["baseline-gs"])),
        "fp32_matches_fig5a": shared_rollout,
        "baseline_above_pm_early": baseline_above_pm_early,
        "sea_dbs_lowest_tail": sea < min(base, pm, gs) - FIG7_MIN_SEPARATION,
        "gs_highest_tail": gs > max(base, pm, sea) + FIG7_MIN_SEPARATION,
        "pm_not_sea": abs(pm - base) < abs(pm - sea),
        "baseline_tail_near_paper": _tail_near("baseline"),
        "pm_tail_near_paper": _tail_near("baseline-pm"),
        "gs_tail_near_paper": _tail_near("baseline-gs"),
        "sea_tail_near_paper": _tail_near("paper"),
    }
    metrics = {
        "tail_means": tail,
        "paper_tail_means": paper_tail,
        "early_mae_baseline": early_mae_b,
        "early_mae_sea": early_mae_s,
        "mae_pm": mae_pm,
        "mae_gs": mae_gs,
        "fp32_vs_fig5a_max_abs": shared_dev,
    }
    notes = [] if have_paper else ["fig7 digitization missing — paper-anchored gates fail closed"]
    return _gate_pack(gates, metrics, notes=notes)


# Table II (paper, PSD units = normalized × 1000): environment seed changed
# every n steps. The Avg Reward column is Eq. (7) of Avg PSD / 1000 in all
# eight cells, so only PSD is gated.
TABLE2_PAPER_PSD: dict[int, tuple[float, float]] = {
    10: (364.0, 326.0),
    20: (366.0, 309.0),
    50: (332.0, 304.0),
    75: (328.0, 302.0),
}
TABLE2_PAPER_REWARD_E2: dict[int, tuple[float, float]] = {
    10: (-1.96, 5.76),
    20: (-2.56, 16.81),
    50: (3.24, 21.16),
    75: (4.84, 23.00),
}
# ±20 PSD units = the ±0.02 normalized level band used on Figs 4a–7.
TABLE2_LEVEL_ABS_TOL = 20.0
TABLE2_GAP_BAND = (0.5, 1.6)


def ravivarapu_table2_gates(rows: dict[int, dict[str, float]]) -> dict[str, Any]:
    """Table II gates. ``rows[n] = {"baseline_psd": …, "sea_psd": …}`` in paper units.

    SEA-DBS below Baseline at every interval, each gap within [0.5, 1.6] ×
    the paper's (38 / 57 / 28 / 26), each cell within ±20 of the paper, and
    average PSD lower at n = 75 than at n = 10 for both variants.
    """
    ns = tuple(TABLE2_PAPER_PSD)
    present = all(n in rows for n in ns)
    if not present:
        return _gate_pack({"all_intervals_present": False}, {"intervals": sorted(rows)})
    b = {n: float(rows[n]["baseline_psd"]) for n in ns}
    s = {n: float(rows[n]["sea_psd"]) for n in ns}
    gap = {n: b[n] - s[n] for n in ns}
    p_gap = {n: TABLE2_PAPER_PSD[n][0] - TABLE2_PAPER_PSD[n][1] for n in ns}
    gates = {
        "all_intervals_present": True,
        "sea_below_baseline_every_n": all(s[n] < b[n] for n in ns),
        "gap_near_paper_every_n": all(_in_band(gap[n], p_gap[n], TABLE2_GAP_BAND) for n in ns),
        "baseline_near_paper_every_n": all(
            abs(b[n] - TABLE2_PAPER_PSD[n][0]) <= TABLE2_LEVEL_ABS_TOL for n in ns
        ),
        "sea_near_paper_every_n": all(
            abs(s[n] - TABLE2_PAPER_PSD[n][1]) <= TABLE2_LEVEL_ABS_TOL for n in ns
        ),
        "psd_falls_with_interval": b[75] < b[10] and s[75] < s[10],
    }
    metrics = {
        "baseline_psd": b,
        "sea_psd": s,
        "gap": gap,
        "paper_gap": p_gap,
        "paper_psd": {n: list(v) for n, v in TABLE2_PAPER_PSD.items()},
    }
    return _gate_pack(gates, metrics)
