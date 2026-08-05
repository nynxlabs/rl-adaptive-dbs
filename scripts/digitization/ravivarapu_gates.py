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


def ravivarapu_fig4a_gates(
    baseline_psd: Sequence[float],
    paper_psd: Sequence[float],
    *,
    n_expected: int = 150,
    early_n: int = 15,
) -> dict[str, Any]:
    """Structural ordering gates (variant labels, decline, SEA-DBS vs Baseline)."""
    b_early = _mean_first_n(baseline_psd, early_n)
    p_early = _mean_first_n(paper_psd, early_n)
    b_arr = _as_fy(baseline_psd)
    p_arr = _as_fy(paper_psd)
    tail = min(30, int(b_arr.size), int(p_arr.size))
    b_late = float(np.mean(b_arr[-tail:])) if tail else float("nan")
    p_late = float(np.mean(p_arr[-tail:])) if tail else float("nan")
    b_drop = b_early - b_late
    p_drop = p_early - p_late
    gates = {
        "shared_start": abs(p_early - b_early) < 0.08,
        "baseline_declines": b_drop > 0.02,
        "paper_declines": p_drop > 0.04,
        "paper_below_baseline_late": p_late < b_late,
        "paper_steeper_drop": p_drop > b_drop,
        "late_gap_min": (b_late - p_late) > 0.01,
        "n_episodes_ok": len(baseline_psd) >= n_expected and len(paper_psd) >= n_expected,
    }
    metrics = {
        "b_early": b_early,
        "b_late": b_late,
        "p_early": p_early,
        "p_late": p_late,
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
    late_lo: float = 120.0,
    drop_frac_of_paper: float = 0.50,
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
    b_late = window_mean(x, b, lo=late_lo)
    s_early = window_mean(x, s, hi=early_hi)
    s_late = window_mean(x, s, lo=late_lo)

    pb_early = window_mean(pbx, pby, hi=early_hi)
    pb_late = window_mean(pbx, pby, lo=late_lo)
    ps_early = window_mean(psx, psy, hi=early_hi)
    ps_late = window_mean(psx, psy, lo=late_lo)

    b_drop = b_early - b_late
    s_drop = s_early - s_late
    pb_drop = pb_early - pb_late
    ps_drop = ps_early - ps_late
    gap = b_late - s_late
    p_gap = pb_late - ps_late

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
        "late_gap_near_paper": rel_close(gap, p_gap, tol=rel_tol + 0.05),
        "late_early_ratio_baseline_near_paper": ratio_close(
            b_late, b_early, pb_late, pb_early, tol=ratio_tol
        ),
        "late_early_ratio_sea_near_paper": ratio_close(
            s_late, s_early, ps_late, ps_early, tol=ratio_tol
        ),
    }
    shape_b = pearson_on_ref_x(pbx, pby, x, b)
    shape_s = pearson_on_ref_x(psx, psy, x, s)

    return _gate_pack(
        gates,
        {
            "b_early": b_early,
            "b_late": b_late,
            "s_early": s_early,
            "s_late": s_late,
            "b_drop": b_drop,
            "s_drop": s_drop,
            "paper_b_early": pb_early,
            "paper_b_late": pb_late,
            "paper_s_early": ps_early,
            "paper_s_late": ps_late,
            "paper_b_drop": pb_drop,
            "paper_s_drop": ps_drop,
            "late_gap": gap,
            "paper_late_gap": p_gap,
            "pearson_baseline": shape_b,
            "pearson_sea": shape_s,
            "n_episodes": n,
        },
        paper_ref={
            "path": str(curves_path("fig4a")),
            "early_hi": early_hi,
            "late_lo": late_lo,
            "source": "refined/fig4a_refined.wpd.tar → curves_fig4a.json",
        },
        notes=["Pearson r is diagnostic only; seeds change wiggles."],
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


def ravivarapu_inference_gates(
    baseline_trace: Sequence[float],
    paper_trace: Sequence[float],
    *,
    carrier_hz: float,
    sea_trace_50hz: Sequence[float] | None = None,
    baseline_trace_50hz: Sequence[float] | None = None,
    n_expected: int = 10,
) -> dict[str, Any]:
    b = _as_fy(baseline_trace)
    p = _as_fy(paper_trace)
    b0, b_end = float(b[0]), float(b[-1])
    p0, p_end = float(p[0]), float(p[-1])
    b_drop = b0 - b_end
    p_drop = p0 - p_end
    gates: dict[str, bool] = {
        "n_steps_ok": int(b.size) == n_expected and int(p.size) == n_expected,
        "shared_start": abs(p0 - b0) < 15.0,
        "baseline_declines": b_drop > 5.0,
        "paper_declines": p_drop > 5.0,
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
    metrics = {
        "carrier_hz": carrier_hz,
        "b_start": b0,
        "b_end": b_end,
        "p_start": p0,
        "p_end": p_end,
        "b_drop": b_drop,
        "p_drop": p_drop,
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
