"""Digitization-anchored gate helpers for Mehregan Paper 1 panels.

Loads WPD-refined ``curves_wpd_refined*.json`` (``n`` + full ``xy``) and
compares replication traces with **x-axis windows** (Hz, seconds, episode
index) — never sample-index fractions.

Design notes
------------
- Paper panels are usually **one RNG realization**. Seeds change wiggles and
  often end levels; gates therefore use ordering, ratios, and relative drops
  with generous tolerances — not pointwise RMSE or wiggle matching.
- Absolute y can differ between plant and paper; prefer ratios to an untreated
  / early baseline when units disagree.
- Fig 5a refined digitization lives under ``artifacts/figures/papers/mehregan/5a/paper_digitization/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Default relative tolerance for paper-level / ratio checks. Wide enough that
# seed-to-seed plant noise does not dominate; tight enough to catch collapse.
DEFAULT_REL_TOL = 0.25
DEFAULT_RATIO_TOL = 0.30
# Mehregan Fig 4a/4b episode-0 level (seed spread of the exploration episode).
EP0_REL_TOL = 0.15

ARTIFACT_ROOT = Path("artifacts/figures/papers/mehregan")


def refined_path(panel: str, *, stem: str = "curves_wpd_refined") -> Path:
    """Canonical refined digitization path for a Mehregan panel."""
    return ARTIFACT_ROOT / panel / "paper_digitization" / f"{stem}.json"


def load_refined(
    path: Path | str,
    *,
    bin_near_duplicates: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from a refined curves JSON.

    Cleans HITL digitization quirks: sort by x, median-bin near-duplicate x
    (covers accidental double passes and small backtracks without dropping
    the early pre-onset segment — important for Fig 2b). Raw per-episode traces
    with one-episode spikes need ``bin_near_duplicates=False``: a spike is
    digitized as top / bottom / top at nearly the same x, and the median erases it.
    """
    payload = json.loads(Path(path).read_text())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, series in payload.get("series", {}).items():
        xy = series.get("xy") or {}
        x = np.asarray(xy.get("x", []), dtype=float)
        y = np.asarray(xy.get("y", []), dtype=float)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            continue
        if bin_near_duplicates:
            x, y = _sort_and_bin_xy(x, y)
        else:
            order = np.argsort(x, kind="mergesort")
            x, y = x[order], y[order]
        out[str(name)] = (x, y)
    return out


def _sort_and_bin_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x; median-combine points that share nearly the same x."""
    if x.size < 2:
        return x, y
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    span = float(np.ptp(x))
    if span <= 0:
        return np.asarray([float(x[0])]), np.asarray([float(np.median(y))])
    # Bin width ~0.25% of axis (or tiny absolute floor).
    width = max(span * 0.0025, 1e-9)
    xs: list[float] = []
    ys: list[float] = []
    i = 0
    n = int(x.size)
    while i < n:
        j = i + 1
        while j < n and (x[j] - x[i]) <= width:
            j += 1
        xs.append(float(np.mean(x[i:j])))
        ys.append(float(np.median(y[i:j])))
        i = j
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def window_mean(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Mean of ``y`` for samples with ``lo <= x <= hi`` (open ends allowed)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    m = np.ones(x.shape, dtype=bool)
    if lo is not None:
        m &= x >= lo
    if hi is not None:
        m &= x <= hi
    if not np.any(m):
        return float("nan")
    return float(np.mean(y[m]))


def rel_close(ours: float, paper: float, tol: float = DEFAULT_REL_TOL) -> bool:
    """True if ``ours`` is within ``tol`` relative to ``|paper|`` (or abs if ~0)."""
    if not np.isfinite(ours) or not np.isfinite(paper):
        return False
    scale = max(abs(paper), 1e-9)
    return abs(ours - paper) / scale <= tol


def ratio_close(
    ours_num: float,
    ours_den: float,
    paper_num: float,
    paper_den: float,
    tol: float = DEFAULT_RATIO_TOL,
) -> bool:
    """Compare ``ours_num/ours_den`` to paper ratio within relative ``tol``."""
    if not all(np.isfinite(v) for v in (ours_num, ours_den, paper_num, paper_den)):
        return False
    if abs(ours_den) < 1e-12 or abs(paper_den) < 1e-12:
        return False
    return rel_close(ours_num / ours_den, paper_num / paper_den, tol=tol)


def pearson_on_ref_x(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    hyp_x: np.ndarray,
    hyp_y: np.ndarray,
) -> float:
    """Pearson r of hyp interpolated onto ref x (diagnostic; not a hard fail)."""
    if ref_x.size < 4 or hyp_x.size < 4:
        return float("nan")
    if float(np.ptp(hyp_x)) <= 0:
        return float("nan")
    hyp_on_ref = np.interp(ref_x, hyp_x, hyp_y)
    if np.std(ref_y) < 1e-12 or np.std(hyp_on_ref) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ref_y, hyp_on_ref)[0, 1])


def pearson_common_grid(
    ref: tuple[np.ndarray, np.ndarray],
    hyp: tuple[np.ndarray, np.ndarray],
    *,
    lo: float | None = None,
    hi: float | None = None,
    n: int = 200,
) -> float:
    """Pearson r of two curves on a uniform grid over the overlap of both x ranges.

    Unlike :func:`pearson_on_ref_x`, the grid does not depend on how densely the
    digitizer clicked one part of the curve, so a dense HITL segment cannot
    dominate the correlation.
    """
    rx, ry = (np.asarray(v, dtype=float) for v in ref)
    hx, hy = (np.asarray(v, dtype=float) for v in hyp)
    if rx.size < 4 or hx.size < 4:
        return float("nan")
    a = max(float(rx.min()), float(hx.min()), -np.inf if lo is None else lo)
    b = min(float(rx.max()), float(hx.max()), np.inf if hi is None else hi)
    if not b > a:
        return float("nan")
    grid = np.linspace(a, b, n)
    r_ = np.interp(grid, rx, ry)
    h_ = np.interp(grid, hx, hy)
    if np.std(r_) < 1e-12 or np.std(h_) < 1e-12:
        return float("nan")
    return float(np.corrcoef(r_, h_)[0, 1])


def boxcar_smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centered ``k``-point running mean with edge padding (``k <= 1`` → copy)."""
    y = np.asarray(y, dtype=float)
    if k <= 1 or y.size < 2:
        return y.copy()
    kernel = np.ones(k) / k
    padded = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _r_at_least(r: float, r_min: float) -> bool:
    return bool(np.isfinite(r) and r >= r_min)


def _ratio_in_band(ours: float, paper: float, lo: float, hi: float) -> bool:
    """``lo <= ours/paper <= hi`` (same sign required; finite, paper ≠ 0)."""
    if not (np.isfinite(ours) and np.isfinite(paper)) or abs(paper) < 1e-12:
        return False
    q = ours / paper
    return bool(lo <= q <= hi)


def _gate_pack(
    gates: dict[str, bool],
    metrics: dict[str, Any],
    *,
    paper_ref: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    bool_gates = {k: bool(v) for k, v in gates.items()}
    return {
        "gates": bool_gates,
        "pass": all(bool_gates.values()) if bool_gates else False,
        "metrics": metrics,
        "paper_ref": paper_ref or {},
        "notes": notes or [],
    }


# ---------------------------------------------------------------------------
# Per-panel builders
# ---------------------------------------------------------------------------


def fig1b_gates(
    replication: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    beta_lo: float = 13.0,
    beta_hi: float = 35.0,
    low_lo: float = 1.0,
    low_hi: float = 12.0,
    shape_lo: float = 1.0,
    shape_hi: float = 50.0,
    smooth_hz: float = 1.5,
    rel_tol: float = 0.30,
    low_beta_tol: float = 0.35,
    peak_tol_hz: float = 4.0,
    pd_shape_r_min: float = 0.6,
    pd130_shape_r_min: float = 0.3,
) -> dict[str, Any]:
    """PSD panel: beta-band ordering, paper-relative levels, spectral shape.

    Shape gates (peak frequency, Pearson r on a common ``shape_lo``–``shape_hi``
    Hz grid) compare a replication series smoothed by a ``smooth_hz`` boxcar
    when it is sampled much finer than the digitization (seed-mean Welch PSD at
    ~0.1 Hz vs ~0.4 Hz digitized points), so bin-level wiggles the paper figure
    cannot show do not decide the correlation. Band means use the raw series.
    """
    paper = paper or load_refined(paper_path or refined_path("1b"))
    keys = ("pd", "healthy", "pd_130hz")
    band = {"beta": (beta_lo, beta_hi), "low": (low_lo, low_hi)}

    def _bands(src: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, float]]:
        return {
            k: {name: window_mean(*src[k], lo=a, hi=b) for name, (a, b) in band.items()}
            for k in keys
            if k in src
        }

    m_ours = _bands(replication)
    m_paper = _bands(paper)
    beta_ours = {k: v["beta"] for k, v in m_ours.items()}
    beta_paper = {k: v["beta"] for k, v in m_paper.items()}

    def _paper_dx(k: str) -> float:
        if k not in paper or paper[k][0].size < 2:
            return float("nan")
        return float(np.median(np.diff(np.sort(paper[k][0]))))

    def _shape_series(k: str) -> tuple[np.ndarray, np.ndarray]:
        x, y = (np.asarray(v, dtype=float) for v in replication[k])
        dx = float(np.median(np.diff(x))) if x.size > 1 else float("nan")
        p_dx = _paper_dx(k)
        if np.isfinite(dx) and dx > 0 and np.isfinite(p_dx) and dx < 0.5 * p_dx:
            return x, boxcar_smooth(y, int(round(smooth_hz / dx)))
        return x, y

    def _peak_hz(x: np.ndarray, y: np.ndarray) -> float:
        m = (x >= shape_lo) & (x <= shape_hi)
        return float(x[m][np.argmax(y[m])]) if np.any(m) else float("nan")

    shaped = {k: _shape_series(k) for k in keys if k in replication}
    shape_r = {
        k: pearson_common_grid(paper[k], shaped[k], lo=shape_lo, hi=shape_hi)
        for k in shaped
        if k in paper
    }
    peak_ours = {k: _peak_hz(*shaped[k]) for k in shaped}
    peak_paper = {k: _peak_hz(*paper[k]) for k in keys if k in paper}
    low_over_beta_ours = {k: v["low"] / v["beta"] for k, v in m_ours.items() if v["beta"]}
    low_over_beta_paper = {k: v["low"] / v["beta"] for k, v in m_paper.items() if v["beta"]}

    gates: dict[str, bool] = {}
    if "pd" in beta_ours and "healthy" in beta_ours:
        gates["pd_gt_healthy"] = beta_ours["pd"] > beta_ours["healthy"]
    if "pd" in beta_ours and "pd_130hz" in beta_ours:
        gates["pd_130_lt_pd"] = beta_ours["pd_130hz"] < beta_ours["pd"]
    if all(k in beta_ours and k in beta_paper for k in ("pd", "pd_130hz")):
        gates["suppression_ratio_near_paper"] = ratio_close(
            beta_ours["pd_130hz"],
            beta_ours["pd"],
            beta_paper["pd_130hz"],
            beta_paper["pd"],
            tol=rel_tol,
        )
    for k in ("healthy", "pd_130hz"):
        if k in beta_ours and k in beta_paper:
            gates[f"{k}_beta_near_paper"] = rel_close(
                beta_ours[k], beta_paper[k], tol=rel_tol
            )
    if all(k in beta_ours and k in beta_paper for k in ("pd", "healthy")):
        gates["healthy_pd_ratio_near_paper"] = ratio_close(
            beta_ours["healthy"],
            beta_ours["pd"],
            beta_paper["healthy"],
            beta_paper["pd"],
            tol=rel_tol,
        )
    if "pd" in beta_ours and "pd" in beta_paper:
        gates["pd_beta_near_paper"] = rel_close(beta_ours["pd"], beta_paper["pd"], tol=rel_tol)
    if "pd" in peak_ours and "pd" in peak_paper:
        gates["pd_peak_freq_near_paper"] = bool(
            np.isfinite(peak_ours["pd"])
            and np.isfinite(peak_paper["pd"])
            and abs(peak_ours["pd"] - peak_paper["pd"]) <= peak_tol_hz
        )
    lob_keys = [k for k in keys if k in low_over_beta_ours and k in low_over_beta_paper]
    if lob_keys:
        gates["low_over_beta_near_paper"] = all(
            rel_close(low_over_beta_ours[k], low_over_beta_paper[k], tol=low_beta_tol)
            for k in lob_keys
        )
    if "pd" in shape_r:
        gates["pd_shape_r"] = _r_at_least(shape_r["pd"], pd_shape_r_min)
    if "pd_130hz" in shape_r:
        gates["pd130_shape_r"] = _r_at_least(shape_r["pd_130hz"], pd130_shape_r_min)
    return _gate_pack(
        gates,
        {
            "beta_ours": beta_ours,
            "beta_paper": beta_paper,
            "low_over_beta_ours": low_over_beta_ours,
            "low_over_beta_paper": low_over_beta_paper,
            "peak_hz_ours": peak_ours,
            "peak_hz_paper": peak_paper,
            "pearson_r": shape_r,
        },
        paper_ref={
            "path": str(paper_path or refined_path("1b")),
            "band_hz": [beta_lo, beta_hi],
            "low_band_hz": [low_lo, low_hi],
            "shape_hz": [shape_lo, shape_hi],
            "smooth_hz": smooth_hz,
        },
        notes=[
            "Shape gates smooth the replication to ~digitization resolution first.",
            "healthy/pd_130 peak frequencies are diagnostic only (flat spectra).",
        ],
    )


def _resolve_series_key(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    primary: str,
    *aliases: str,
) -> str | None:
    """Return the first matching key in ``series``."""
    for name in (primary, *aliases):
        if name in series:
            return name
    return None


def half_drop_time(
    x: np.ndarray,
    y: np.ndarray,
    *,
    onset: float = 2.0,
    late_lo: float = 6.0,
    t_end: float = 12.0,
    n: int = 501,
) -> float:
    """First time after ``onset`` the trace reaches half its pre→late drop.

    Pre = mean over ``t <= onset``; late = mean over ``t >= late_lo``; the
    trace is linearly interpolated on ``n`` points over ``[onset, t_end]``.
    NaN if it never crosses (or there is no drop).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pre = window_mean(x, y, hi=onset)
    late = window_mean(x, y, lo=late_lo)
    if not (np.isfinite(pre) and np.isfinite(late)) or late >= pre:
        return float("nan")
    thr = pre - 0.5 * (pre - late)
    grid = np.linspace(onset, t_end, n)
    yy = np.interp(grid, x, y)
    below = np.flatnonzero(yy <= thr)
    return float(grid[below[0]]) if below.size else float("nan")


def fig2_time_gates(
    replication: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "2a",
    onset: float = 2.0,
    late_lo: float = 6.0,
    mid_lo: float = 4.0,
    mid_hi: float = 7.0,
    t_end: float = 12.0,
    untreated_key: str = "pd",
    treated_key: str = "pd_130hz",
    rep_untreated: str | None = None,
    rep_treated: str | None = None,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    baseline_rel_tol: float = 0.05,
    untreated_flat_abs_tol: float = 0.15,
    prestim_level_tol: float = 0.15,
    t50_tol_s: float = 1.0,
    shape_r_min: float = 0.7,
) -> dict[str, Any]:
    """Fig 2a/2b: shared baseline, suppression levels/timing, treated shape.

    Pearson r is gated on the treated trace and on the untreated − treated gap
    only: the untreated trace is flat noise around its pre-onset level, so its
    correlation with the paper's (one realization) is meaningless.
    """
    paper = paper or load_refined(paper_path or refined_path(panel))
    ru = rep_untreated or untreated_key
    rt = rep_treated or treated_key
    if ru not in replication or rt not in replication:
        return _gate_pack(
            {"series_present": False},
            {},
            notes=[f"missing replication keys {ru!r}/{rt!r}"],
        )
    paper_u_key = _resolve_series_key(paper, untreated_key, "PD no Treatment", "PD no stim")
    paper_t_key = _resolve_series_key(
        paper, treated_key, "PD 130Hz Treatment", "PD + 130Hz cDBS"
    )
    if paper_u_key is None or paper_t_key is None:
        return _gate_pack(
            {"paper_series_present": False},
            {},
            notes=[f"missing paper keys {untreated_key!r}/{treated_key!r}"],
        )
    o_u, o_t = replication[ru], replication[rt]
    p_u, p_t = paper[paper_u_key], paper[paper_t_key]

    def _levels(u: tuple[np.ndarray, np.ndarray], t: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        return {
            "untreated_pre": window_mean(*u, hi=onset),
            "treated_pre": window_mean(*t, hi=onset),
            "untreated_late": window_mean(*u, lo=late_lo),
            "treated_late": window_mean(*t, lo=late_lo),
            "untreated_mid": window_mean(*u, lo=mid_lo, hi=mid_hi),
            "treated_mid": window_mean(*t, lo=mid_lo, hi=mid_hi),
            "treated_t50": half_drop_time(*t, onset=onset, late_lo=late_lo, t_end=t_end),
        }

    o = _levels(o_u, o_t)
    p = _levels(p_u, p_t)
    for d in (o, p):
        d["late_ratio"] = (
            d["treated_late"] / d["untreated_late"] if abs(d["untreated_late"]) > 1e-12 else float("nan")
        )
        d["untreated_late_over_pre"] = d["untreated_late"] / d["untreated_pre"]
        d["treated_late_over_pre"] = d["treated_late"] / d["treated_pre"]

    grid = np.linspace(0.0, t_end, 121)

    def _gap(u: tuple[np.ndarray, np.ndarray], t: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.interp(grid, *u) - np.interp(grid, *t)

    gap_o, gap_p = _gap(o_u, o_t), _gap(p_u, p_t)
    gap_r = (
        float(np.corrcoef(gap_o, gap_p)[0, 1])
        if np.std(gap_o) > 1e-12 and np.std(gap_p) > 1e-12
        else float("nan")
    )
    treated_r = pearson_common_grid(p_t, o_t, lo=0.0, hi=t_end)

    shared = abs(o["untreated_pre"] - o["treated_pre"]) / max(abs(o["untreated_pre"]), 1e-9) <= baseline_rel_tol
    gates = {
        "prestim_shared": shared,
        "treated_below_untreated_late": o["treated_late"] < o["untreated_late"],
        "late_ratio_near_paper": ratio_close(
            o["treated_late"], o["untreated_late"], p["treated_late"], p["untreated_late"], tol=ratio_tol
        ),
        "suppression_drop_near_paper": ratio_close(
            o["untreated_late"] - o["treated_late"],
            max(abs(o["untreated_pre"]), 1e-9),
            p["untreated_late"] - p["treated_late"],
            max(abs(p["untreated_pre"]), 1e-9),
            tol=ratio_tol,
        ),
        "untreated_flat": bool(
            np.isfinite(o["untreated_late_over_pre"])
            and abs(o["untreated_late_over_pre"] - p["untreated_late_over_pre"]) <= untreated_flat_abs_tol
        ),
        "treated_late_over_pre_near_paper": ratio_close(
            o["treated_late"], o["treated_pre"], p["treated_late"], p["treated_pre"], tol=ratio_tol
        ),
        "prestim_level_near_paper": rel_close(
            o["untreated_pre"], p["untreated_pre"], tol=prestim_level_tol
        ),
        "treated_mid_ratio_near_paper": ratio_close(
            o["treated_mid"], o["untreated_mid"], p["treated_mid"], p["untreated_mid"], tol=ratio_tol
        ),
        "t50_near_paper": bool(
            np.isfinite(o["treated_t50"])
            and np.isfinite(p["treated_t50"])
            and abs(o["treated_t50"] - p["treated_t50"]) <= t50_tol_s
        ),
        "treated_shape_r": _r_at_least(treated_r, shape_r_min),
        "gap_shape_r": _r_at_least(gap_r, shape_r_min),
    }
    return _gate_pack(
        gates,
        {
            "ours": o,
            "paper": p,
            "treated_pearson_r": treated_r,
            "gap_pearson_r": gap_r,
        },
        paper_ref={
            "path": str(paper_path or refined_path(panel)),
            "onset": onset,
            "late_lo": late_lo,
            "mid_window": [mid_lo, mid_hi],
            "shape_grid": [0.0, t_end],
        },
        notes=[
            "Seed changes post-onset floor; ratio/drop gates absorb that.",
            "Untreated-only Pearson r is not gated (flat noise).",
        ],
    )


def fig4a_gates(
    beta_trace: list[float] | np.ndarray,
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    early_hi: float = 130.0,
    late_lo: float = 150.0,
    drop_frac_of_paper: float = 0.70,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    n_expected: int | None = 300,
    steps_per_episode: int = 30,
    n_episodes: int = 10,
    episode_shape_r_min: float = 0.7,
    late_level_tol: float = 0.15,
    episode_drop_band: tuple[float, float] = (0.6, 1.6),
    rebound_margin: float = 0.02,
    half_drop_tol_ep: int = 1,
    per_episode_tol: float = 0.20,
    mid_fade_band: tuple[float, float] = (0.5, 2.0),
) -> dict[str, Any]:
    """Training beta vs step: trend, drop, per-episode trajectory vs paper.

    Episode-level gates use the 30-step episode means of the replication and
    of the digitized paper (window means over steps ``[30 i, 30 i + 30)``) —
    the paper is one realization with step-level noise, so the trajectory is
    compared at episode resolution.
    """
    paper = paper or load_refined(paper_path or refined_path("4a"))
    y = np.asarray(beta_trace, dtype=float)
    x = np.arange(y.size, dtype=float)
    # Paper digitization x is training step (0..~300).
    train_key = "training" if "training" in paper else next(iter(paper))
    px, py = paper[train_key]
    early = window_mean(x, y, hi=early_hi)
    late = window_mean(x, y, lo=late_lo)
    p_early = window_mean(px, py, hi=early_hi)
    p_late = window_mean(px, py, lo=late_lo)
    drop = early - late
    p_drop = p_early - p_late
    start_w = float(np.mean(y[: min(30, y.size)])) if y.size else float("nan")
    end_w = float(np.mean(y[max(0, y.size - 30) :])) if y.size else float("nan")
    # Paper digitization: early[0,100]≈0.49, mid[120,150]≈0.445 (modest mid fade,
    # not a vertical cliff). Require mid below early by a paper-scaled margin.
    early_lo = window_mean(x, y, hi=100.0)
    mid = window_mean(x, y, lo=120.0, hi=150.0)
    p_early_lo = window_mean(px, py, hi=100.0)
    p_mid = window_mean(px, py, lo=120.0, hi=150.0)
    p_mid_drop = p_early_lo - p_mid
    mid_drop = early_lo - mid
    ep0 = window_mean(x, y, hi=29.0)
    p_ep0 = window_mean(px, py, hi=29.0)

    spe = int(steps_per_episode)
    n_ep = min(int(n_episodes), y.size // spe)
    ep_o = np.array([float(np.mean(y[i * spe : (i + 1) * spe])) for i in range(n_ep)])
    ep_p = np.array(
        [window_mean(px, py, lo=i * spe, hi=(i + 1) * spe - 1e-9) for i in range(n_ep)]
    )

    def _half_drop_ep(ep: np.ndarray) -> int | None:
        """First episode whose mean reaches half the eps 0–3 → last-4 drop."""
        if ep.size < 8:
            return None
        hi_, lo_ = float(np.mean(ep[:4])), float(np.mean(ep[-4:]))
        hit = np.flatnonzero(ep <= hi_ - 0.5 * (hi_ - lo_))
        return int(hit[0]) if hit.size else None

    full = n_ep == int(n_episodes) and n_ep >= 10
    ep_r = (
        float(np.corrcoef(ep_o, ep_p)[0, 1])
        if n_ep >= 4 and np.std(ep_o) > 1e-12 and np.std(ep_p) > 1e-12
        else float("nan")
    )
    late_ep = float(np.mean(ep_o[5:])) if full else float("nan")
    p_late_ep = float(np.mean(ep_p[5:])) if full else float("nan")
    ep_drop = float(np.mean(ep_o[:4]) - np.mean(ep_o[-4:])) if full else float("nan")
    p_ep_drop = float(np.mean(ep_p[:4]) - np.mean(ep_p[-4:])) if full else float("nan")
    rebound = float(np.mean(ep_o[7:10]) - np.mean(ep_o[5:7])) if full else float("nan")
    half_o, half_p = _half_drop_ep(ep_o), _half_drop_ep(ep_p)
    ep_rel_err = (ep_o - ep_p) / np.abs(ep_p) if n_ep else np.array([])
    gates = {
        "plot_style": n_expected is None or y.size == n_expected,
        "overall_trend_down": end_w < start_w,
        "drop_vs_paper": bool(
            np.isfinite(drop) and np.isfinite(p_drop) and drop >= drop_frac_of_paper * p_drop
        ),
        "late_early_ratio_near_paper": ratio_close(late, early, p_late, p_early, tol=ratio_tol),
        # Match paper's mid-training fade (digitized ~0.047): two-sided, so a
        # cliff (fade ≫ paper) fails as well as no fade.
        "mid_fade_vs_paper": bool(
            np.isfinite(p_mid_drop)
            and p_mid_drop > 0
            and _ratio_in_band(mid_drop, p_mid_drop, *mid_fade_band)
        ),
        # Phase 1 (digitization revisit): first episode should sit near untreated /
        # paper ep0 (~0.50), not already suppressed (~0.44 on τ→1.4).
        # Episode 0 is exploration over near-zero logits: its mean is a seed
        # draw (seeds 0-9 span 0.43-0.55, mean 0.47; probe
        # scripts/probes/mehregan_ep0_seeds.py). 15% keeps 9/10 seeds in band.
        "ep0_near_paper": rel_close(ep0, p_ep0, tol=EP0_REL_TOL),
        "episode_shape_r": _r_at_least(ep_r, episode_shape_r_min),
        "late_level_near_paper": full and rel_close(late_ep, p_late_ep, tol=late_level_tol),
        "episode_drop_near_paper": full and _ratio_in_band(ep_drop, p_ep_drop, *episode_drop_band),
        # Eps 7–9 must not climb back above eps 5–6 (the paper settles, it does
        # not rebound once the policy has found the suppressing pattern).
        "no_late_rebound": bool(full and np.isfinite(rebound) and rebound <= rebound_margin),
        "half_drop_episode_near_paper": bool(
            half_o is not None and half_p is not None and abs(half_o - half_p) <= half_drop_tol_ep
        ),
        "per_episode_near_paper": bool(
            full and np.all(np.isfinite(ep_rel_err)) and np.max(np.abs(ep_rel_err)) <= per_episode_tol
        ),
    }
    return _gate_pack(
        gates,
        {
            "episode_means": ep_o.tolist(),
            "paper_episode_means": ep_p.tolist(),
            "episode_rel_err": ep_rel_err.tolist(),
            "episode_pearson_r": ep_r,
            "late_eps5_9": late_ep,
            "paper_late_eps5_9": p_late_ep,
            "episode_drop": ep_drop,
            "paper_episode_drop": p_ep_drop,
            "late_rebound": rebound,
            "half_drop_episode": half_o,
            "paper_half_drop_episode": half_p,
            "early_mean": early,
            "late_mean": late,
            "drop": drop,
            "early_0_100": early_lo,
            "mid_120_150": mid,
            "mid_drop": mid_drop,
            "paper_early": p_early,
            "paper_late": p_late,
            "paper_drop": p_drop,
            "paper_early_0_100": p_early_lo,
            "paper_mid_120_150": p_mid,
            "paper_mid_drop": p_mid_drop,
            "ep0_mean": ep0,
            "paper_ep0": p_ep0,
            "late_early_ratio": late / early if abs(early) > 1e-12 else float("nan"),
            "paper_late_early_ratio": p_late / p_early if abs(p_early) > 1e-12 else float("nan"),
        },
        paper_ref={"path": str(paper_path or refined_path("4a")), "early_hi": early_hi, "late_lo": late_lo},
        notes=[
            "Absolute early/late bands removed; seed changes level.",
            "mid_fade_vs_paper: fade within 0.5–2x the digitized paper mid fade.",
            "ep0_near_paper is the phase-1 digitization target (untreated-like start).",
        ],
    )


# Fig 4b: the paper draws 10 vertices (one per training episode, matching Fig
# 4a's 10 x 30 steps) evenly across a 0–8 x axis, so vertex i sits at x = 8i/9.
FIG4B_EPISODES = 10
FIG4B_PAPER_X_MAX = 8.0


def fig4b_paper_x_to_episode(x: np.ndarray, *, n_episodes: int = FIG4B_EPISODES) -> np.ndarray:
    """Map Fig 4b paper x (0–8 axis) to episode index (0 … ``n_episodes`` − 1)."""
    return np.asarray(x, dtype=float) * (n_episodes - 1) / FIG4B_PAPER_X_MAX


def fig4b_paper_per_episode(
    xy: tuple[np.ndarray, np.ndarray],
    n: int,
    *,
    n_episodes: int = FIG4B_EPISODES,
) -> np.ndarray:
    """Digitized Fig 4b curve interpolated at episodes ``0 … n-1``.

    Interpolating at the episode positions (instead of window means over the
    clicked points) keeps a stray click between vertices from biasing a window
    and fills a vertex the digitizer skipped from its neighbours.
    """
    x, y = xy
    return np.interp(np.arange(n, dtype=float), fig4b_paper_x_to_episode(x, n_episodes=n_episodes), y)


def fig4b_gates(
    episode_rewards: list[float] | np.ndarray,
    episode_mean_beta: list[float] | np.ndarray,
    *,
    paper_reward: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_psd: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    reward_path: Path | str | None = None,
    psd_path: Path | str | None = None,
    early_n: int = 3,
    late_start: int = 6,
    n_expected: int | None = FIG4B_EPISODES,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    late_beta_rel_tol: float = 0.15,
    reward_threshold: float = 0.35,
    late_reward_hi: float = 2.0,
    shape_r_min: float = 0.7,
    early_reward_tol: float = 0.50,
    reward_floor_factor: float = 1.5,
    reward_max: float = 5.0,
    psd_floor_factor: float = 0.9,
    per_episode_psd_tol: float = 0.20,
) -> dict[str, Any]:
    """Reward rise + episode PSD drop vs paper digitizations, per episode.

    Paper per-episode values are the digitized curves interpolated at episode
    ``i`` after mapping paper x → ``9 x / 8`` (:func:`fig4b_paper_per_episode`).
    Paper late episode-mean PSD sits just above the reward threshold
    ``β_t = 0.35`` (~0.37) so episode reward approaches 0 from below.
    ``late_beta_near_paper``, ``late_reward_near_zero`` and ``max_reward_le_cap``
    reject a collapse onto a deep suppressor (late PSD ~0.30, reward ~+15).
    The reward scale matches the paper (digitized ep 0 ≈ −81), so magnitude
    gates (early level, floor) apply as well as the qualitative rise.
    """
    paper_reward = paper_reward or load_refined(
        reward_path or refined_path("4b", stem="curves_wpd_refined_reward")
    )
    paper_psd = paper_psd or load_refined(
        psd_path or refined_path("4b", stem="curves_wpd_refined_psd")
    )
    r = np.asarray(episode_rewards, dtype=float)
    b = np.asarray(episode_mean_beta, dtype=float)
    n = int(r.size)
    if n < 2 or b.size != n:
        return _gate_pack({"enough_episodes": False}, {"n": n, "n_beta": int(b.size)})

    rp = fig4b_paper_per_episode(paper_reward[next(iter(paper_reward))], n)
    bp = fig4b_paper_per_episode(paper_psd[next(iter(paper_psd))], n)
    e_hi = min(early_n, n)
    l_lo = min(late_start, n - 1)
    early_r, p_early_r = float(np.mean(r[:e_hi])), float(np.mean(rp[:e_hi]))
    late_r, p_late_r = float(np.mean(r[l_lo:])), float(np.mean(rp[l_lo:]))
    early_b, p_early_b = float(np.mean(b[:e_hi])), float(np.mean(bp[:e_hi]))
    late_b, p_late_b = float(np.mean(b[l_lo:])), float(np.mean(bp[l_lo:]))

    def _r(a: np.ndarray, c: np.ndarray) -> float:
        if a.size < 4 or np.std(a) < 1e-12 or np.std(c) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, c)[0, 1])

    reward_r, psd_r = _r(r, rp), _r(b, bp)
    psd_rel_err = (b - bp) / np.abs(bp)

    rise_episode = None
    for i in range(1, n):
        if r[i] > r[0] + 10.0:
            rise_episode = i
            break

    gates = {
        "plot_style": n_expected is None or n == n_expected,
        "early_negative": early_r < 0.0,
        "reward_rises": late_r > early_r,
        "late_plateau_improved": late_r > -10.0,
        "rise_timing": rise_episode is not None and rise_episode <= min(6, n - 1),
        "beta_drops": late_b < early_b,
        "beta_drop_ratio_near_paper": ratio_close(
            late_b, early_b, p_late_b, p_early_b, tol=ratio_tol
        ),
        "reward_recovers_like_paper": bool(
            late_r > early_r and p_late_r > p_early_r and late_r > -10.0
        ),
        "late_beta_near_paper": rel_close(late_b, p_late_b, tol=late_beta_rel_tol),
        "late_reward_near_zero": bool(
            np.isfinite(late_r) and late_r > -10.0 and late_r <= late_reward_hi
        ),
        # Phase 1: match episode 0 to digitized paper (~0.50 PSD) before chasing
        # the late floor.
        "ep0_beta_near_paper": rel_close(float(b[0]), float(bp[0]), tol=EP0_REL_TOL),
        "reward_shape_r": _r_at_least(reward_r, shape_r_min),
        "psd_shape_r": _r_at_least(psd_r, shape_r_min),
        "early_reward_near_paper": rel_close(early_r, p_early_r, tol=early_reward_tol),
        # Not more negative than 1.5x the paper's worst episode (paper ep 0 ≈ −81).
        "reward_floor_not_deeper": bool(float(r.min()) >= reward_floor_factor * float(rp.min())),
        # Paper reward never goes positive (≤ −1.7): a run that over-suppresses
        # below β_t earns large positive episodes the figure does not show.
        "max_reward_le_cap": bool(float(r.max()) <= reward_max),
        "psd_floor_not_deeper": bool(float(b.min()) >= psd_floor_factor * float(bp.min())),
        "per_episode_psd_near_paper": bool(
            np.all(np.isfinite(psd_rel_err)) and np.max(np.abs(psd_rel_err)) <= per_episode_psd_tol
        ),
    }
    return _gate_pack(
        gates,
        {
            "n_episodes": n,
            "early_reward": early_r,
            "late_reward": late_r,
            "early_beta": early_b,
            "late_beta": late_b,
            "rise_episode": rise_episode,
            "paper_episode_rewards": rp.tolist(),
            "paper_episode_mean_beta": bp.tolist(),
            "paper_early_reward": p_early_r,
            "paper_late_reward": p_late_r,
            "paper_early_beta": p_early_b,
            "paper_late_beta": p_late_b,
            "ep0_beta": float(b[0]),
            "paper_ep0_beta": float(bp[0]),
            "reward_pearson_r": reward_r,
            "psd_pearson_r": psd_r,
            "min_reward": float(r.min()),
            "paper_min_reward": float(rp.min()),
            "max_reward": float(r.max()),
            "min_beta": float(b.min()),
            "paper_min_beta": float(bp.min()),
            "psd_rel_err": psd_rel_err.tolist(),
            "reward_threshold": reward_threshold,
            "late_beta_above_threshold": bool(np.isfinite(late_b) and late_b >= reward_threshold),
            "late_beta_rel_tol": late_beta_rel_tol,
        },
        paper_ref={
            "reward": str(reward_path or refined_path("4b", stem="curves_wpd_refined_reward")),
            "psd": str(psd_path or refined_path("4b", stem="curves_wpd_refined_psd")),
            "paper_x_to_episode": f"episode = x * {FIG4B_EPISODES - 1} / {FIG4B_PAPER_X_MAX:g}",
        },
        notes=[
            "Paper per-episode values: digitization interpolated at x = 8 i / 9.",
        ],
    )


FIG5_PAPER_SERIES: dict[str, tuple[str, ...]] = {
    "no_stim": ("PD no stim", "pd", "PD no Treatment"),
    "trained": ("Fully Trained 30Hz", "Fully Trained 45Hz", "trained"),
    "periodic": ("Periodic 30Hz", "Periodic 45Hz", "periodic"),
    "cdbs_130": ("Periodic 130Hz", "cdbs_130"),
}


def fig5_efficacy_gates(
    panel_means: dict[str, float],
    *,
    traces: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "5b",
    onset: float = 2.0,
    early_hi: float = 4.0,
    late_lo: float = 4.0,
    t_end: float = 12.0,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    require_cdbs: bool = False,
    skip_paper_ratios: bool = False,
    prestim_spread_tol: float = 0.03,
    prestim_level_tol: float = 0.10,
    shape_r_min: float = 0.6,
    gap_band: tuple[float, float] = (0.4, 2.0),
) -> dict[str, Any]:
    """Fig 5a/5b post-onset means: ordering, paper ratios, trace-level gates.

    ``panel_means`` holds late (t ≥ ``late_lo``) means per series. ``traces``
    (series → ``(time_s, P_beta)``, display axis with onset at ``onset``) adds
    the trace gates: shared pre-onset level (relative spread across **all**
    series), pre-onset level vs paper, trained onset speed on
    ``[onset, early_hi]``, trained / periodic shape r, the trained − periodic
    gap and (5a) the 130 Hz ratio.

    ``skip_paper_ratios=True`` only when digitization is missing or marked bad.
    """
    no_stim = panel_means.get("no_stim")
    trained = panel_means.get("trained")
    periodic = panel_means.get("periodic")
    cdbs = panel_means.get("cdbs_130")

    if trained is None or no_stim is None or periodic is None:
        return _gate_pack(
            {"means_present": False},
            {"panel_means": panel_means},
            notes=["missing trained/no_stim/periodic means"],
        )

    gates: dict[str, bool] = {
        "trained_below_no_stim": trained < no_stim,
    }
    if panel == "5b":
        gates["trained_below_periodic"] = trained < periodic
        gates["periodic_above_no_stim"] = periodic > no_stim
    else:
        gates["trained_above_periodic"] = trained > periodic
        if require_cdbs and cdbs is not None:
            gates["cdbs_lowest"] = cdbs < trained and cdbs < periodic and cdbs < no_stim

    notes = [
        "Post-onset means vary by seed; ratio gates use paper late window t>=4.",
    ]
    paper_metrics: dict[str, Any] = {}
    trace_metrics: dict[str, Any] = {}
    if skip_paper_ratios:
        notes.append("paper_digitization needs_redo — qualitative ordering only.")
    else:
        paper = paper or load_refined(paper_path or refined_path(panel))
        p_xy = {
            k: next((paper[a] for a in aliases if a in paper), None)
            for k, aliases in FIG5_PAPER_SERIES.items()
        }

        def _paper_late(k: str) -> float:
            xy = p_xy.get(k)
            return window_mean(*xy, lo=late_lo) if xy is not None else float("nan")

        p_ns, p_tr, p_pe, p_cd = (_paper_late(k) for k in ("no_stim", "trained", "periodic", "cdbs_130"))
        paper_metrics = {"no_stim": p_ns, "trained": p_tr, "periodic": p_pe, "cdbs_130": p_cd}
        if all(np.isfinite(v) for v in (p_ns, p_tr, p_pe)):
            gates["trained_no_stim_ratio_near_paper"] = ratio_close(
                trained, no_stim, p_tr, p_ns, tol=ratio_tol
            )
            gates["periodic_no_stim_ratio_near_paper"] = ratio_close(
                periodic, no_stim, p_pe, p_ns, tol=ratio_tol
            )
            # (trained − periodic)/no-stim late: 5a > 0 (periodic 45 Hz beats the
            # policy), 5b < 0 (periodic 30 Hz elevates beta). Same sign, 0.4–2x.
            gap = (trained - periodic) / no_stim
            p_gap = (p_tr - p_pe) / p_ns
            gates["trained_periodic_gap_near_paper"] = _ratio_in_band(gap, p_gap, *gap_band)
            paper_metrics["trained_periodic_gap"] = p_gap
            trace_metrics["trained_periodic_gap"] = gap
        if panel == "5a" and require_cdbs:
            gates["cdbs_no_stim_ratio_near_paper"] = bool(
                cdbs is not None
                and np.isfinite(p_cd)
                and ratio_close(cdbs, no_stim, p_cd, p_ns, tol=ratio_tol)
            )

        if traces is not None:
            series = [k for k in FIG5_PAPER_SERIES if k in traces]
            pre = {k: window_mean(*traces[k], hi=onset) for k in series}
            p_pre = {
                k: window_mean(*p_xy[k], hi=onset) for k in series if p_xy.get(k) is not None
            }
            pre_vals = np.array(list(pre.values()), dtype=float)
            level = float(np.mean(pre_vals)) if pre_vals.size else float("nan")
            p_level = float(np.mean(list(p_pre.values()))) if p_pre else float("nan")
            gates["shared_baseline"] = bool(
                pre_vals.size >= 3
                and np.all(np.isfinite(pre_vals))
                and float(np.ptp(pre_vals)) <= prestim_spread_tol * abs(level)
            )
            gates["prestim_level_near_paper"] = rel_close(level, p_level, tol=prestim_level_tol)

            def _early(xy: tuple[np.ndarray, np.ndarray] | None) -> float:
                return window_mean(*xy, lo=onset, hi=early_hi) if xy is not None else float("nan")

            gates["trained_onset_ratio_near_paper"] = ratio_close(
                _early(traces.get("trained")),
                _early(traces.get("no_stim")),
                _early(p_xy.get("trained")),
                _early(p_xy.get("no_stim")),
                tol=ratio_tol,
            )
            shape_r: dict[str, float] = {}
            for k in ("trained", "periodic"):
                if k in traces and p_xy.get(k) is not None:
                    shape_r[k] = pearson_common_grid(p_xy[k], traces[k], lo=0.0, hi=t_end)
                gates[f"{k}_shape_r"] = _r_at_least(shape_r.get(k, float("nan")), shape_r_min)
            trace_metrics.update(
                {
                    "pre": pre,
                    "paper_pre": p_pre,
                    "pre_spread_rel": float(np.ptp(pre_vals) / abs(level)) if pre_vals.size else float("nan"),
                    "pearson_r": shape_r,
                }
            )

    return _gate_pack(
        gates,
        {"ours": panel_means, "paper_late": paper_metrics, "trace": trace_metrics},
        paper_ref={
            "path": None if skip_paper_ratios else str(paper_path or refined_path(panel)),
            "late_lo": late_lo,
            "onset_window": [onset, early_hi],
            "skip_paper_ratios": skip_paper_ratios,
        },
        notes=notes,
    )


def fig6_quant_gates(
    traces: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    qat_seeds: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "6a",
    onset: float = 2.0,
    late_lo: float = 4.0,
    level_tol: float = 0.20,
    closed_loop: dict[str, bool] | None = None,
    early_hi: float = 4.0,
    t_end: float = 12.0,
    onset_tol: float = 0.20,
    prestim_level_tol: float = 0.10,
    shape_r_min: float = 0.6,
    qat_gap_frac: float = 0.5,
    ptq_alter_frac: float = 0.5,
) -> dict[str, Any]:
    """Fig 6a/6b: PTQ tracks fp32 suppression; 10-episode QAT does not.

    ``traces`` maps ``fp32`` / ``ptq_fp16`` / ``ptq_int8`` to ``(time_s,
    P_beta)`` on the paper display axis (onset at 2 s). Levels are compared as
    late (t ≥ ``late_lo``) mean over the shared pre-onset mean, the same
    normalization applied to the digitized paper curves — plant and paper
    absolute PSD scales differ slightly, the pre-stim baseline does not move.

    QAT is judged as a **seed band**: ``qat_seeds`` maps each QAT training seed
    to its eval trace (without it, ``traces["qat"]`` is a one-seed band).
    10-episode int8 QAT does not converge, so the pattern its deployed argmax
    plays varies by training seed; the QAT gates test the paper's claim on the
    share of seeds and on the median seed, not on one draw.

    ``closed_loop`` (series → bool; ``qat`` = every QAT seed) records that every
    series ran its own (quantized) actor closed loop; any replayed / open-loop
    series fails.

    §IV.A.3: "quantization induces some alterations in RL model outputs, the
    overall DBS efficacy remains unaffected" — ``ptq_*_near_fp32`` test the
    second half, ``ptq_*_alters_trace`` the first: mean |PTQ − fp32| over
    ``t >= onset`` must reach ``ptq_alter_frac`` of the digitized paper's
    (well above the 1.6–3.9 mean |Δ| between the same curve digitized in two
    panels, e.g. Fig 5a trained vs Fig 6a fp32).
    """
    paper = paper or load_refined(paper_path or refined_path(panel))
    alias = {
        "fp32": ("Fully Trained 45Hz", "Fully Trained 30Hz", "fp32"),
        "ptq_int8": ("PTQ, INT8", "ptq_int8"),
        "ptq_fp16": ("PTQ, FP16", "ptq_fp16"),
        "qat": ("QAT", "qat"),
    }

    def _paper_xy(key: str) -> tuple[np.ndarray, np.ndarray] | None:
        for a in alias[key]:
            if a in paper:
                return paper[a]
        return None

    def _rel(xy: tuple[np.ndarray, np.ndarray] | None) -> tuple[float, float, float]:
        if xy is None:
            return float("nan"), float("nan"), float("nan")
        x, y = (np.asarray(v, dtype=float) for v in xy)
        pre = window_mean(x, y, hi=onset)
        late = window_mean(x, y, lo=late_lo)
        return pre, late, late / pre if pre else float("nan")

    band = dict(qat_seeds) if qat_seeds else ({"0": traces["qat"]} if "qat" in traces else {})
    band_rel = {str(s): _rel(xy) for s, xy in band.items()}
    seed_pre = np.array([v[0] for v in band_rel.values()], dtype=float)
    seed_late = np.array([v[1] for v in band_rel.values()], dtype=float)
    seed_rel = np.array([v[2] for v in band_rel.values()], dtype=float)

    def _median(v: np.ndarray) -> float:
        return float(np.median(v)) if v.size else float("nan")

    keys = ("fp32", "ptq_fp16", "ptq_int8", "qat")
    ours = {k: _rel(traces.get(k)) for k in keys[:3]}
    ours["qat"] = (_median(seed_pre), _median(seed_late), _median(seed_rel))
    theirs = {k: _rel(_paper_xy(k)) for k in keys}
    o_pre = {k: ours[k][0] for k in keys}
    o_late = {k: ours[k][1] for k in keys}
    o_rel = {k: ours[k][2] for k in keys}
    p_rel = {k: theirs[k][2] for k in keys}

    gates: dict[str, bool] = {}
    if closed_loop is not None:
        gates["all_closed_loop"] = all(bool(closed_loop.get(k)) for k in keys)
    pres = [o_pre[k] for k in keys[:3]] + seed_pre.tolist()
    finite_pre = [v for v in pres if np.isfinite(v)]
    # Same seed + same reset → one shared pre-onset trace. 1% of the level
    # (≈5 PSD units) absorbs digitization spread on the paper's own curves.
    gates["prestim_shared"] = bool(
        seed_pre.size
        and len(finite_pre) == len(pres)
        and (max(finite_pre) - min(finite_pre)) <= 0.01 * abs(np.mean(finite_pre))
    )
    gates["fp32_suppresses_vs_baseline"] = bool(o_late["fp32"] < o_pre["fp32"])
    gates["ptq_fp16_near_fp32"] = rel_close(o_late["ptq_fp16"], o_late["fp32"], tol=0.15)
    gates["ptq_int8_near_fp32"] = rel_close(o_late["ptq_int8"], o_late["fp32"], tol=0.20)
    gates["qat_elevated_vs_fp32"] = bool(o_late["qat"] > o_late["fp32"])
    for k in ("fp32", "ptq_fp16", "ptq_int8"):
        gates[f"{k}_level_near_paper"] = rel_close(o_rel[k], p_rel[k], tol=level_tol)
    # §IV.A.3: under QAT the power "stayed at the same range or increased" and the
    # model "failed to generate a stimulation pattern capable of reducing" it —
    # a claim about the method, so a strict majority of QAT training seeds must
    # not drop below the digitized paper band (one-sided, as the text) ...
    paper_like = (
        seed_rel >= (1.0 - level_tol) * p_rel["qat"]
        if np.isfinite(p_rel["qat"])
        else np.zeros(seed_rel.shape, dtype=bool)
    )
    gates["qat_majority_not_below_paper"] = bool(
        seed_rel.size and 2 * int(np.sum(paper_like)) > seed_rel.size
    )
    # ... and the median seed must sit in the figure's "same range" (two-sided).
    gates["qat_median_near_paper"] = rel_close(o_rel["qat"], p_rel["qat"], tol=level_tol)

    if band:
        qt = np.asarray(next(iter(band.values()))[0], dtype=float)
        qy = np.median(np.vstack([np.asarray(xy[1], dtype=float) for xy in band.values()]), axis=0)
        early_post = window_mean(qt, qy, lo=onset, hi=8.0)
        late_post = window_mean(qt, qy, lo=10.0, hi=12.0)
        # Paper QAT stays in its elevated band to the end (no late fade into the
        # suppressed fp32 band); allow a 10% sag vs the early post-onset mean.
        gates["qat_late_sustained"] = bool(
            np.isfinite(late_post) and np.isfinite(early_post) and late_post >= 0.90 * early_post
        )

    # Digitization-anchored onset / shape / level gates on fp32, the QAT gap,
    # and the PTQ alteration magnitude.
    p_fp32 = _paper_xy("fp32")
    o_fp32 = traces.get("fp32")

    def _early_over_pre(xy: tuple[np.ndarray, np.ndarray] | None) -> float:
        if xy is None:
            return float("nan")
        pre = window_mean(*xy, hi=onset)
        return window_mean(*xy, lo=onset, hi=early_hi) / pre if pre else float("nan")

    o_onset, p_onset = _early_over_pre(o_fp32), _early_over_pre(p_fp32)
    gates["fp32_onset_near_paper"] = rel_close(o_onset, p_onset, tol=onset_tol)
    fp32_r = (
        pearson_common_grid(p_fp32, o_fp32, lo=0.0, hi=t_end)
        if p_fp32 is not None and o_fp32 is not None
        else float("nan")
    )
    gates["fp32_shape_r"] = _r_at_least(fp32_r, shape_r_min)
    gates["prestim_level_near_paper"] = rel_close(o_pre["fp32"], theirs["fp32"][0], tol=prestim_level_tol)
    o_qat_gap = (o_late["qat"] - o_late["fp32"]) / o_pre["fp32"] if o_pre["fp32"] else float("nan")
    p_qat_gap = (
        (theirs["qat"][1] - theirs["fp32"][1]) / theirs["fp32"][0] if theirs["fp32"][0] else float("nan")
    )
    gates["qat_fp32_gap_vs_paper"] = bool(
        np.isfinite(o_qat_gap) and np.isfinite(p_qat_gap) and p_qat_gap > 0
        and o_qat_gap >= qat_gap_frac * p_qat_gap
    )

    def _mean_abs_diff(
        a: tuple[np.ndarray, np.ndarray] | None, b: tuple[np.ndarray, np.ndarray] | None
    ) -> float:
        if a is None or b is None:
            return float("nan")
        lo_ = max(onset, float(np.min(a[0])), float(np.min(b[0])))
        hi_ = min(t_end, float(np.max(a[0])), float(np.max(b[0])))
        if not hi_ > lo_:
            return float("nan")
        grid = np.linspace(lo_, hi_, 201)
        return float(np.mean(np.abs(np.interp(grid, *a) - np.interp(grid, *b))))

    alter_ours: dict[str, float] = {}
    alter_paper: dict[str, float] = {}
    for k in ("ptq_int8", "ptq_fp16"):
        alter_ours[k] = _mean_abs_diff(traces.get(k), o_fp32)
        alter_paper[k] = _mean_abs_diff(_paper_xy(k), p_fp32)
        gates[f"{k}_alters_trace"] = bool(
            np.isfinite(alter_ours[k])
            and np.isfinite(alter_paper[k])
            and alter_ours[k] >= ptq_alter_frac * alter_paper[k]
        )
    return _gate_pack(
        gates,
        {
            "ours_pre": o_pre,
            "ours_late": o_late,
            "ours_late_over_pre": o_rel,
            "paper_late_over_pre": p_rel,
            "paper_pre": {k: theirs[k][0] for k in keys},
            "paper_late": {k: theirs[k][1] for k in keys},
            "qat_seed_late_over_pre": {s: v[2] for s, v in band_rel.items()},
            "qat_seed_paper_like": {s: bool(v) for s, v in zip(band_rel, paper_like)},
            "qat_paper_like_count": int(np.sum(paper_like)),
            "qat_seed_count": int(seed_rel.size),
            "fp32_onset_over_pre": o_onset,
            "paper_fp32_onset_over_pre": p_onset,
            "fp32_pearson_r": fp32_r,
            "qat_fp32_gap": o_qat_gap,
            "paper_qat_fp32_gap": p_qat_gap,
            "ptq_mean_abs_diff": alter_ours,
            "paper_ptq_mean_abs_diff": alter_paper,
        },
        paper_ref={
            "path": str(paper_path or refined_path(panel)),
            "late_lo": late_lo,
            "level_tol": level_tol,
        },
        notes=[
            "Paper: PTQ (fp16, int8) tracks fp32 suppression; 10-episode QAT stays near pre-stim.",
            "Levels are late/pre ratios so plant-vs-paper PSD scale offsets cancel.",
            "QAT is a seed band: ours_*['qat'] are medians over the QAT training seeds.",
        ],
    )
