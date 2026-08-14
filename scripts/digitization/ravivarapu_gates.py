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
    "shared_start": "shape",
    "baseline_declines": "shape",
    "paper_declines": "shape",
    "paper_below_baseline_late": "shape",
    "paper_steeper_drop": "shape",
    "late_gap_min": "shape",
    "final_window_gap_substantial": "shape",
    "n_episodes_ok": "shape",
    "dig_enough_episodes": "shape",
    "dig_sea_steeper_than_baseline_like_paper": "shape",
    "dig_sea_below_baseline_late_like_paper": "shape",
    "dig_progressive_decline_baseline": "shape",
    "dig_progressive_decline_sea": "shape",
    "dig_gradual_decline_baseline": "shape",
    "dig_gradual_decline_sea": "shape",
    "dig_gap_widens_mid_to_late": "shape",
    "dig_early_mid_to_mid_drop_sea_not_front_loaded": "shape",
    "dig_drop_timing_baseline": "shape",
    "dig_drop_timing_sea": "shape",
    "dig_pearson_baseline_min": "shape",
    "dig_pearson_sea_min": "shape",
    "dig_shared_start_near_paper": "full",
    "dig_baseline_drop_vs_paper": "full",
    "dig_sea_drop_vs_paper": "full",
    "dig_late_gap_near_paper": "full",
    "dig_final_window_gap_near_paper": "full",
    "dig_late_early_ratio_baseline_near_paper": "full",
    "dig_late_early_ratio_sea_near_paper": "full",
    "dig_early_mid_baseline_near_paper": "full",
    "dig_early_mid_sea_near_paper": "full",
}


def ravivarapu_fig4a_tier_pass(flat: dict[str, Any], *, full: bool) -> bool:
    """Return whether every gate in the shape or full tier passes."""
    tiers = ("shape", "full") if full else ("shape",)
    for key, tier in RAVIVARAPU_FIG4A_GATE_TIER.items():
        if tier not in tiers:
            continue
        value = flat.get(key)
        if isinstance(value, bool) and not value:
            return False
    return True


def ravivarapu_fig4a_attach_tiered_pass(flat: dict[str, Any]) -> dict[str, Any]:
    """Set ``shape_pass`` (phase 1) and ``pass`` (ship) on a flattened Fig 4a manifest."""
    out = dict(flat)
    out["shape_pass"] = ravivarapu_fig4a_tier_pass(out, full=False)
    out["pass"] = ravivarapu_fig4a_tier_pass(out, full=True)
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
    paper_psd: Sequence[float],
    *,
    n_expected: int = 150,
    early_n: int = 15,
    final_window_gap_min: float = 0.03,
) -> dict[str, Any]:
    """Structural ordering gates for the Baseline/SEA-DBS training curves.

    The final-window check is intentionally absolute on the normalized Fig 4a
    scale. A late ordering pass with only a barely perceptible separation is not
    enough for this panel.
    """
    b_early = _mean_first_n(baseline_psd, early_n)
    p_early = _mean_first_n(paper_psd, early_n)
    b_arr = _as_fy(baseline_psd)
    p_arr = _as_fy(paper_psd)
    tail = min(30, int(b_arr.size), int(p_arr.size))
    b_late = float(np.mean(b_arr[-tail:])) if tail else float("nan")
    p_late = float(np.mean(p_arr[-tail:])) if tail else float("nan")
    final_tail = min(10, int(b_arr.size), int(p_arr.size))
    b_final = float(np.mean(b_arr[-final_tail:])) if final_tail else float("nan")
    p_final = float(np.mean(p_arr[-final_tail:])) if final_tail else float("nan")
    b_drop = b_early - b_late
    p_drop = p_early - p_late
    final_gap = b_final - p_final
    gates = {
        "shared_start": abs(p_early - b_early) < 0.08,
        "baseline_declines": b_drop > 0.02,
        "paper_declines": p_drop > 0.04,
        "paper_below_baseline_late": p_late < b_late,
        "paper_steeper_drop": p_drop > b_drop,
        "late_gap_min": (b_late - p_late) > 0.01,
        "final_window_gap_substantial": bool(
            np.isfinite(final_gap) and final_gap >= final_window_gap_min
        ),
        "n_episodes_ok": len(baseline_psd) >= n_expected and len(paper_psd) >= n_expected,
    }
    metrics = {
        "b_early": b_early,
        "b_late": b_late,
        "p_early": p_early,
        "p_late": p_late,
        "b_final": b_final,
        "p_final": p_final,
        "final_window_gap": final_gap,
        "final_window_gap_min": final_window_gap_min,
        "b_drop": b_drop,
        "p_drop": p_drop,
        "n_baseline": len(baseline_psd),
        "n_paper": len(paper_psd),
    }
    return _gate_pack(gates, metrics)


def ravivarapu_fig4a_digitization_gates(
    baseline_psd: Sequence[float],
    sea_psd: Sequence[float],
    *,
    early_hi: float = 15.0,
    early_mid_lo: float = 15.0,
    early_mid_hi: float = 40.0,
    mid_lo: float = 40.0,
    mid_hi: float = 80.0,
    late_lo: float = 120.0,
    final_window_lo: float = 140.0,
    timing_episode: float = 50.0,
    drop_frac_of_paper: float = 0.40,
    profile_frac_of_paper: float = 0.30,
    early_mid_rel_tol: float = 0.12,
    pearson_baseline_min: float = 0.58,
    pearson_sea_min: float = 0.58,
    rel_tol: float = DEFAULT_REL_TOL,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    n_expected: int = 150,
) -> dict[str, Any]:
    """Fig 4a episode PSD vs refined WPD curves (``curves_fig4a.json``)."""
    paper = load_curves("fig4a")
    pbx, pby = paper["Baseline"]
    psx, psy = paper["SEA-DBS"]

    n = min(len(baseline_psd), len(sea_psd))
    if n < 50:
        return _gate_pack(
            {"enough_episodes": False},
            {"n_episodes": n},
            paper_ref={"path": str(curves_path("fig4a"))},
        )

    x = np.arange(n, dtype=float)
    b = _as_fy(baseline_psd[:n])
    s = _as_fy(sea_psd[:n])

    b_early = window_mean(x, b, hi=early_hi)
    b_early_mid = window_mean(x, b, lo=early_mid_lo, hi=early_mid_hi)
    b_mid = window_mean(x, b, lo=mid_lo, hi=mid_hi)
    b_late = window_mean(x, b, lo=late_lo)
    b_final = window_mean(x, b, lo=final_window_lo)
    s_early = window_mean(x, s, hi=early_hi)
    s_early_mid = window_mean(x, s, lo=early_mid_lo, hi=early_mid_hi)
    s_mid = window_mean(x, s, lo=mid_lo, hi=mid_hi)
    s_late = window_mean(x, s, lo=late_lo)
    s_final = window_mean(x, s, lo=final_window_lo)

    pb_early = window_mean(pbx, pby, hi=early_hi)
    pb_early_mid = window_mean(pbx, pby, lo=early_mid_lo, hi=early_mid_hi)
    pb_mid = window_mean(pbx, pby, lo=mid_lo, hi=mid_hi)
    pb_late = window_mean(pbx, pby, lo=late_lo)
    pb_final = window_mean(pbx, pby, lo=final_window_lo)
    ps_early = window_mean(psx, psy, hi=early_hi)
    ps_early_mid = window_mean(psx, psy, lo=early_mid_lo, hi=early_mid_hi)
    ps_mid = window_mean(psx, psy, lo=mid_lo, hi=mid_hi)
    ps_late = window_mean(psx, psy, lo=late_lo)
    ps_final = window_mean(psx, psy, lo=final_window_lo)

    # The original level gates average the last 30 episodes. That can hide a
    # late convergence: v39's final points were nearly identical although the
    # 30-episode means remained separated. The final ten-episode window is the
    # hard check; the single endpoint is retained as a diagnostic because it
    # is one noisy plant realization rather than a stable curve feature.
    b_end = _interp_at(x, b, n - 1)
    s_end = _interp_at(x, s, n - 1)
    pb_end = _interp_at(pbx, pby, n - 1)
    ps_end = _interp_at(psx, psy, n - 1)

    b_drop = b_early - b_late
    s_drop = s_early - s_late
    pb_drop = pb_early - pb_late
    ps_drop = ps_early - ps_late
    gap = b_late - s_late
    p_gap = pb_late - ps_late
    final_gap = b_final - s_final
    paper_final_gap = pb_final - ps_final
    endpoint_gap = b_end - s_end
    paper_endpoint_gap = pb_end - ps_end
    # Gradual-decline profile (paper declines in EVERY window): mid (40-80) must
    # sit above late (120-150) by a meaningful fraction of the paper's own
    # mid-late drop, so a fast-then-flat step does not pass the shape check.
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

    gap_mid = b_mid - s_mid
    gap_midlate = window_mean(x, b - s, lo=80.0, hi=120.0)
    gap_late_window = b_late - s_late

    gates = {
        "enough_episodes": n >= n_expected,
        "shared_start_near_paper": rel_close(s_early, ps_early, tol=rel_tol)
        and rel_close(b_early, pb_early, tol=rel_tol),
        "baseline_drop_vs_paper": bool(
            np.isfinite(b_drop) and np.isfinite(pb_drop) and b_drop >= drop_frac_of_paper * pb_drop
        ),
        "sea_drop_vs_paper": bool(
            np.isfinite(s_drop) and np.isfinite(ps_drop) and s_drop >= drop_frac_of_paper * ps_drop
        ),
        "sea_steeper_than_baseline_like_paper": bool(s_drop > b_drop and ps_drop > pb_drop),
        "sea_below_baseline_late_like_paper": bool(s_late < b_late and ps_late < pb_late),
        # The substantial final-window gap is the hard contrast target. A
        # larger late gap should not fail as "too far" from a noisy digitized
        # paper estimate, so this paper check is one-sided.
        "late_gap_near_paper": bool(
            np.isfinite(gap)
            and np.isfinite(p_gap)
            and p_gap > 0
            and gap >= (1.0 - (rel_tol + 0.05)) * p_gap
        ),
        "final_window_gap_near_paper": bool(
            np.isfinite(final_gap)
            and np.isfinite(paper_final_gap)
            and paper_final_gap > 0
            and final_gap >= 0.50 * paper_final_gap
        ),
        "early_mid_to_mid_drop_sea_not_front_loaded": bool(
            np.isfinite(s_early_mid)
            and np.isfinite(s_mid)
            and np.isfinite(ps_early_mid)
            and np.isfinite(ps_mid)
            and (s_early_mid - s_mid) <= 1.75 * (ps_early_mid - ps_mid)
        ),
        "late_early_ratio_baseline_near_paper": ratio_close(
            b_late, b_early, pb_late, pb_early, tol=ratio_tol
        ),
        "late_early_ratio_sea_near_paper": ratio_close(
            s_late, s_early, ps_late, ps_early, tol=ratio_tol
        ),
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
        "progressive_decline_baseline": _nonincreasing_window_means(
            b_early, b_early_mid, b_mid, b_late, b_final
        ),
        "progressive_decline_sea": _nonincreasing_window_means(
            s_early, s_early_mid, s_mid, s_late, s_final
        ),
        "gap_widens_mid_to_late": _nondecreasing_gaps(
            gap_mid, gap_midlate, gap_late_window
        ),
        "early_mid_baseline_near_paper": rel_close(
            b_early_mid, pb_early_mid, tol=early_mid_rel_tol
        ),
        "early_mid_sea_near_paper": rel_close(
            s_early_mid, ps_early_mid, tol=early_mid_rel_tol
        ),
        "drop_timing_baseline": _drop_timing_ok(b_drop_frac, pb_drop_frac),
        "drop_timing_sea": _drop_timing_ok(s_drop_frac, ps_drop_frac),
        "pearson_baseline_min": bool(
            np.isfinite(shape_b) and shape_b >= pearson_baseline_min
        ),
        "pearson_sea_min": bool(np.isfinite(shape_s) and shape_s >= pearson_sea_min),
    }

    return _gate_pack(
        gates,
        {
            "b_early": b_early,
            "b_early_mid": b_early_mid,
            "b_mid": b_mid,
            "b_late": b_late,
            "s_early": s_early,
            "s_early_mid": s_early_mid,
            "s_mid": s_mid,
            "s_late": s_late,
            "b_final": b_final,
            "s_final": s_final,
            "b_end": b_end,
            "s_end": s_end,
            "b_drop": b_drop,
            "s_drop": s_drop,
            "b_midlate": b_midlate,
            "s_midlate": s_midlate,
            "b_drop_frac_at_timing": b_drop_frac,
            "s_drop_frac_at_timing": s_drop_frac,
            "paper_b_early": pb_early,
            "paper_b_early_mid": pb_early_mid,
            "paper_b_mid": pb_mid,
            "paper_b_late": pb_late,
            "paper_s_early": ps_early,
            "paper_s_early_mid": ps_early_mid,
            "paper_s_mid": ps_mid,
            "paper_s_late": ps_late,
            "paper_b_final": pb_final,
            "paper_s_final": ps_final,
            "paper_b_end": pb_end,
            "paper_s_end": ps_end,
            "paper_b_drop": pb_drop,
            "paper_s_drop": ps_drop,
            "paper_b_midlate": pb_midlate,
            "paper_s_midlate": ps_midlate,
            "paper_b_drop_frac_at_timing": pb_drop_frac,
            "paper_s_drop_frac_at_timing": ps_drop_frac,
            "gap_mid": gap_mid,
            "gap_midlate": gap_midlate,
            "gap_late_window": gap_late_window,
            "late_gap": gap,
            "paper_late_gap": p_gap,
            "final_window_gap": final_gap,
            "paper_final_window_gap": paper_final_gap,
            "endpoint_gap": endpoint_gap,
            "paper_endpoint_gap": paper_endpoint_gap,
            "pearson_baseline": shape_b,
            "pearson_sea": shape_s,
            "pearson_baseline_min": pearson_baseline_min,
            "pearson_sea_min": pearson_sea_min,
            "timing_episode": timing_episode,
            "n_episodes": n,
        },
        paper_ref={
            "path": str(curves_path("fig4a")),
            "early_hi": early_hi,
            "early_mid_lo": early_mid_lo,
            "early_mid_hi": early_mid_hi,
            "mid_lo": mid_lo,
            "mid_hi": mid_hi,
            "late_lo": late_lo,
            "final_window_lo": final_window_lo,
            "timing_episode": timing_episode,
            "source": "refined/fig4a_refined.wpd.tar → curves_fig4a.json",
        },
        notes=[
            "Shape gates catch front-loaded drops that match early/late anchors but not the paper trajectory.",
            f"Pearson r minima: baseline>={pearson_baseline_min}, SEA>={pearson_sea_min} (seed-0 ship).",
        ],
    )


def ravivarapu_fig4b_gates(
    baseline_reward: Sequence[float],
    paper_reward: Sequence[float],
) -> dict[str, Any]:
    b_early, b_late = _early_late(baseline_reward)
    p_early, p_late = _early_late(paper_reward)
    n = min(len(baseline_reward), len(paper_reward))
    mid_lo = min(40, max(0, n // 4))
    mid_hi = min(80, n)
    gates = {
        "paper_above_baseline_late": p_late > b_late,
        "paper_pull_ahead_mid": _mid_mean(paper_reward, mid_lo, mid_hi)
        > _mid_mean(baseline_reward, mid_lo, mid_hi),
        "both_rise": (p_late - p_early) > 0.3 and (b_late - b_early) > 0.3,
    }
    metrics = {
        "b_early": b_early,
        "b_late": b_late,
        "p_early": p_early,
        "p_late": p_late,
        "n_episodes": n,
    }
    return _gate_pack(gates, metrics)


# Fig 5 shape only: any net drop counts. Digitized paper drops (~0.10/0.15 at
# 50 Hz, ~0.06/0.07 at 30 Hz) are reference, not gate thresholds.
INFERENCE_DECLINE_MIN = 0.0
INFERENCE_SHARED_START_MAX = 0.05
# Fig 5a steps 0–5 vs digitized paper (norm = crop PSD / 1000). Generous:
# match the early window "to an extent", not Fig 4a-style magnitude polish.
INFERENCE_EARLY_N = 6
INFERENCE_EARLY_MAE_MAX = 0.030
INFERENCE_EARLY_SEA_MAE_3_5_MAX = 0.015
INFERENCE_EARLY_SEA_DROP_MIN = 0.050
INFERENCE_PAPER_Y_TO_NORM = 1000.0
INFERENCE_EARLY_SERIES_50HZ = ("Baseline 50Hz", "SEA-DBS 50Hz")


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
    digitized paper with generous MAE / drop tols (not 10-step polish).

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
    if carrier_hz == 50.0 and b.size >= INFERENCE_EARLY_N and p.size >= INFERENCE_EARLY_N:
        paper_b = _paper_early_norm("fig5a", INFERENCE_EARLY_SERIES_50HZ[0])
        paper_s = _paper_early_norm("fig5a", INFERENCE_EARLY_SERIES_50HZ[1])
        b_early = b[:INFERENCE_EARLY_N]
        p_early = p[:INFERENCE_EARLY_N]
        if paper_b is not None and paper_s is not None:
            early_mae_b = float(np.mean(np.abs(b_early - paper_b)))
            early_mae_p = float(np.mean(np.abs(p_early - paper_s)))
            early_mae_p_35 = float(np.mean(np.abs(p_early[3:] - paper_s[3:])))
            early_drop_p = float(p_early[0] - p_early[-1])
            early_drop_b = float(b_early[0] - b_early[-1])
            gates["early_mae_baseline"] = early_mae_b <= INFERENCE_EARLY_MAE_MAX
            gates["early_mae_sea"] = early_mae_p <= INFERENCE_EARLY_MAE_MAX
            gates["early_mae_sea_3_5"] = early_mae_p_35 <= INFERENCE_EARLY_SEA_MAE_3_5_MAX
            gates["early_sea_declines"] = early_drop_p > INFERENCE_EARLY_SEA_DROP_MIN
            gates["early_baseline_declines"] = early_drop_b > INFERENCE_DECLINE_MIN
            # Paper: Baseline above SEA from step 1 through 5, not overlaid.
            gates["early_sea_below_baseline"] = bool(
                np.all(p_early[1:] < b_early[1:])
            )
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
    }
    return _gate_pack(gates, metrics)


def ravivarapu_fig6_gates(traces: dict[str, Sequence[float]]) -> dict[str, Any]:
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

    shared_start = (
        abs(float(b[0]) - float(sea[0])) < 20.0
        and abs(float(b_ptq[0]) - float(sea_ptq[0])) < 20.0
    )
    gates = {
        "four_series_present": True,
        "shared_start": shared_start,
        "sea_below_baseline": sea_late < b_late,
        "sea_ptq_below_baseline": sea_ptq_late < b_late,
        "sea_ptq_tracks_fp32": rel_close(sea_ptq_late, sea_fp_late, tol=DEFAULT_REL_TOL + 0.15),
        "baseline_ptq_near_or_above_baseline": b_ptq_late >= b_late * 0.95,
    }
    metrics = {
        "b_late": b_late,
        "b_ptq_late": b_ptq_late,
        "sea_late": sea_late,
        "sea_ptq_late": sea_ptq_late,
    }
    return _gate_pack(gates, metrics)


def ravivarapu_fig7_gates(traces: dict[str, Sequence[float]]) -> dict[str, Any]:
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

    gates = {
        "four_variants_present": True,
        "sea_dbs_lowest_tail": sea <= min(tail.values()) + 1e-6,
        "gs_highest_or_near_highest_tail": gs >= max(tail.values()) - 5.0,
        "pm_not_sea": abs(pm - base) <= abs(pm - sea),
        "shared_start": abs(float(_as_fy(traces["baseline"])[0]) - float(_as_fy(traces["paper"])[0])) < 20.0,
        "n_steps_ok": all(len(traces[k]) == 10 for k in required),
    }
    return _gate_pack(gates, {"tail_means": tail})
