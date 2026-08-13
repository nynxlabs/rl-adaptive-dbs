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

ARTIFACT_ROOT = Path("artifacts/figures/papers/mehregan")


def refined_path(panel: str, *, stem: str = "curves_wpd_refined") -> Path:
    """Canonical refined digitization path for a Mehregan panel."""
    return ARTIFACT_ROOT / panel / "paper_digitization" / f"{stem}.json"


def load_refined(path: Path | str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from a refined curves JSON.

    Cleans HITL digitization quirks: sort by x, median-bin near-duplicate x
    (covers accidental double passes and small backtracks without dropping
    the early pre-onset segment — important for Fig 2b).
    """
    payload = json.loads(Path(path).read_text())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, series in payload.get("series", {}).items():
        xy = series.get("xy") or {}
        x = np.asarray(xy.get("x", []), dtype=float)
        y = np.asarray(xy.get("y", []), dtype=float)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            continue
        x, y = _sort_and_bin_xy(x, y)
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
    rel_tol: float = 0.30,
) -> dict[str, Any]:
    """PSD panel: beta-band ordering + paper-relative suppression ratios.

    Absolute PD beta can sit above the paper (plant scale / seed). Prefer
    ordering plus ``pd_130/pd`` and healthy level near paper.
    """
    paper = paper or load_refined(paper_path or refined_path("1b"))
    keys = ("pd", "healthy", "pd_130hz")
    beta_ours = {
        k: window_mean(*replication[k], lo=beta_lo, hi=beta_hi)
        for k in keys
        if k in replication
    }
    beta_paper = {
        k: window_mean(*paper[k], lo=beta_lo, hi=beta_hi) for k in keys if k in paper
    }
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
    shape: dict[str, float] = {}
    for k in keys:
        if k in replication and k in paper:
            shape[k] = pearson_on_ref_x(*paper[k], *replication[k])
    return _gate_pack(
        gates,
        {
            "beta_ours": beta_ours,
            "beta_paper": beta_paper,
            "pearson_r": shape,
        },
        paper_ref={"path": str(paper_path or refined_path("1b")), "band_hz": [beta_lo, beta_hi]},
        notes=[
            "Pearson r is diagnostic only; seeds change PSD wiggles.",
            "PD absolute level is not hard-gated (plant/seed scale).",
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


def fig2_time_gates(
    replication: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "2a",
    onset: float = 2.0,
    late_lo: float = 6.0,
    untreated_key: str = "pd",
    treated_key: str = "pd_130hz",
    rep_untreated: str | None = None,
    rep_treated: str | None = None,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    baseline_rel_tol: float = 0.05,
) -> dict[str, Any]:
    """Fig 2a/2b: shared pre-onset baseline, post-onset blue-below-red, late ratio."""
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

    o_u_pre = window_mean(*replication[ru], hi=onset)
    o_t_pre = window_mean(*replication[rt], hi=onset)
    o_u_late = window_mean(*replication[ru], lo=late_lo)
    o_t_late = window_mean(*replication[rt], lo=late_lo)
    p_u_late = window_mean(*paper[paper_u_key], lo=late_lo)
    p_t_late = window_mean(*paper[paper_t_key], lo=late_lo)
    p_u_pre = window_mean(*paper[paper_u_key], hi=onset)
    p_t_pre = window_mean(*paper[paper_t_key], hi=onset)

    shared = abs(o_u_pre - o_t_pre) / max(abs(o_u_pre), 1e-9) <= baseline_rel_tol
    gates = {
        "prestim_shared": shared,
        "treated_below_untreated_late": o_t_late < o_u_late,
        "late_ratio_near_paper": ratio_close(
            o_t_late, o_u_late, p_t_late, p_u_late, tol=ratio_tol
        ),
        "suppression_drop_near_paper": ratio_close(
            o_u_late - o_t_late,
            max(abs(o_u_pre), 1e-9),
            p_u_late - p_t_late,
            max(abs(p_u_pre), 1e-9),
            tol=ratio_tol,
        ),
    }
    return _gate_pack(
        gates,
        {
            "ours": {
                "untreated_pre": o_u_pre,
                "treated_pre": o_t_pre,
                "untreated_late": o_u_late,
                "treated_late": o_t_late,
                "late_ratio": o_t_late / o_u_late if abs(o_u_late) > 1e-12 else float("nan"),
            },
            "paper": {
                "untreated_pre": p_u_pre,
                "treated_pre": p_t_pre,
                "untreated_late": p_u_late,
                "treated_late": p_t_late,
                "late_ratio": p_t_late / p_u_late if abs(p_u_late) > 1e-12 else float("nan"),
            },
        },
        paper_ref={"path": str(paper_path or refined_path(panel)), "onset": onset, "late_lo": late_lo},
        notes=["Seed changes post-onset floor; ratio/drop gates absorb that."],
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
) -> dict[str, Any]:
    """Training beta vs step: trend down, drop vs paper, late/early ratio."""
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
    gates = {
        "plot_style": n_expected is None or y.size == n_expected,
        "overall_trend_down": end_w < start_w,
        "drop_vs_paper": bool(
            np.isfinite(drop) and np.isfinite(p_drop) and drop >= drop_frac_of_paper * p_drop
        ),
        "late_early_ratio_near_paper": ratio_close(late, early, p_late, p_early, tol=ratio_tol),
        # Match paper's mid-training fade (digitized ~0.04), allow 50% of paper mid-drop.
        "mid_fade_vs_paper": bool(
            np.isfinite(mid_drop)
            and np.isfinite(p_mid_drop)
            and p_mid_drop > 0
            and mid_drop >= 0.50 * p_mid_drop
        ),
        # Phase 1 (digitization revisit): first episode should sit near untreated /
        # paper ep0 (~0.50), not already suppressed (~0.44 on τ→1.4).
        "ep0_near_paper": rel_close(ep0, p_ep0, tol=0.10),
    }
    return _gate_pack(
        gates,
        {
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
            "mid_fade_vs_paper uses digitized paper mid fade (not a hard cliff).",
            "ep0_near_paper is the phase-1 digitization target (untreated-like start).",
        ],
    )


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
    ratio_tol: float = DEFAULT_RATIO_TOL,
    late_beta_rel_tol: float = 0.15,
    reward_threshold: float = 0.35,
    late_reward_hi: float = 2.0,
) -> dict[str, Any]:
    """Reward rise + episode PSD drop vs paper digitizations.

    Digitization revisit (Report 3): paper late episode-mean PSD sits *above*
    reward threshold ``β_t = 0.35`` (~0.37) so episode reward approaches 0
    from below. A full collapse onto one suppressing pattern drives late PSD
    ~0.30 and flips reward positive — reject that floor even when the
    late/early *ratio* still looks paper-like.
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
    if n < 2 or b.size < 2:
        return _gate_pack({"enough_episodes": False}, {"n": n})

    early_r = float(np.mean(r[: min(early_n, n)]))
    late_r = float(np.mean(r[min(late_start, n - 1) :]))
    early_b = float(np.mean(b[: min(early_n, b.size)]))
    late_b = float(np.mean(b[min(late_start, b.size - 1) :]))

    rk = next(iter(paper_reward))
    pk = next(iter(paper_psd))
    rx, ry = paper_reward[rk]
    px, py = paper_psd[pk]
    # Paper x is episode index ~0..8
    p_early_r = window_mean(rx, ry, hi=float(early_n - 1) + 0.5)
    p_late_r = window_mean(rx, ry, lo=float(late_start) - 0.5)
    p_early_b = window_mean(px, py, hi=float(early_n - 1) + 0.5)
    p_late_b = window_mean(px, py, lo=float(late_start) - 0.5)
    p_ep0_b = window_mean(px, py, lo=-0.5, hi=0.5)

    rise_episode = None
    for i in range(1, n):
        if r[i] > r[0] + 10.0:
            rise_episode = i
            break

    gates = {
        "early_negative": early_r < 0.0,
        "reward_rises": late_r > early_r,
        "late_plateau_improved": late_r > -10.0,
        "rise_timing": rise_episode is not None and rise_episode <= min(6, n - 1),
        "beta_drops": late_b < early_b,
        "beta_drop_ratio_near_paper": ratio_close(
            late_b, early_b, p_late_b, p_early_b, tol=ratio_tol
        ),
        # Reward absolute scale differs from paper (~-80 vs ~-30); require the
        # same qualitative recovery (rise + late improved), not magnitude match.
        "reward_recovers_like_paper": bool(
            late_r > early_r and p_late_r > p_early_r and late_r > -10.0
        ),
        # Paper late PSD ~0.37 stays above β_t so Eq. (8) never enters the
        # positive linear branch. Locked v4/v18 collapse to ~0.30 and go to +16.
        "late_beta_above_threshold": bool(np.isfinite(late_b) and late_b >= reward_threshold),
        "late_beta_near_paper": rel_close(late_b, p_late_b, tol=late_beta_rel_tol),
        "late_reward_near_zero": bool(
            np.isfinite(late_r) and late_r > -10.0 and late_r <= late_reward_hi
        ),
        # Phase 1: match episode 0 to digitized paper (~0.50 PSD) before chasing
        # the late floor. v30 ep0≈0.44 already suppresses vs untreated ~0.47.
        "ep0_beta_near_paper": rel_close(float(b[0]), p_ep0_b, tol=0.10),
    }
    return _gate_pack(
        gates,
        {
            "early_reward": early_r,
            "late_reward": late_r,
            "early_beta": early_b,
            "late_beta": late_b,
            "rise_episode": rise_episode,
            "paper_early_reward": p_early_r,
            "paper_late_reward": p_late_r,
            "paper_early_beta": p_early_b,
            "paper_late_beta": p_late_b,
            "ep0_beta": float(b[0]),
            "paper_ep0_beta": p_ep0_b,
            "reward_threshold": reward_threshold,
            "late_beta_rel_tol": late_beta_rel_tol,
        },
        paper_ref={
            "reward": str(reward_path or refined_path("4b", stem="curves_wpd_refined_reward")),
            "psd": str(psd_path or refined_path("4b", stem="curves_wpd_refined_psd")),
        },
    )


def fig5_efficacy_gates(
    panel_means: dict[str, float],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "5b",
    late_lo: float = 4.0,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    require_cdbs: bool = False,
    skip_paper_ratios: bool = False,
) -> dict[str, Any]:
    """Fig 5a/5b post-onset means: ordering + optional paper ratios.

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
    if skip_paper_ratios:
        notes.append("paper_digitization needs_redo — qualitative ordering only.")
    else:
        paper = paper or load_refined(paper_path or refined_path(panel))
        # Map paper series names
        name_map = {
            "no_stim": ("PD no stim", "pd", "PD no Treatment"),
            "trained": (
                "Fully Trained 30Hz",
                "Fully Trained 45Hz",
                "trained",
            ),
            "periodic": ("Periodic 30Hz", "Periodic 45Hz", "periodic"),
        }

        def _paper_late(aliases: tuple[str, ...]) -> float:
            for a in aliases:
                if a in paper:
                    return window_mean(*paper[a], lo=late_lo)
            return float("nan")

        p_ns = _paper_late(name_map["no_stim"])
        p_tr = _paper_late(name_map["trained"])
        p_pe = _paper_late(name_map["periodic"])
        paper_metrics = {"no_stim": p_ns, "trained": p_tr, "periodic": p_pe}
        if all(np.isfinite(v) for v in (p_ns, p_tr, p_pe)):
            gates["trained_no_stim_ratio_near_paper"] = ratio_close(
                trained, no_stim, p_tr, p_ns, tol=ratio_tol
            )
            gates["periodic_no_stim_ratio_near_paper"] = ratio_close(
                periodic, no_stim, p_pe, p_ns, tol=ratio_tol
            )

    return _gate_pack(
        gates,
        {"ours": panel_means, "paper_late": paper_metrics},
        paper_ref={
            "path": None if skip_paper_ratios else str(paper_path or refined_path(panel)),
            "late_lo": late_lo,
            "skip_paper_ratios": skip_paper_ratios,
        },
        notes=notes,
    )


def fig6_quant_gates(
    post_means: dict[str, float],
    *,
    paper: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    paper_path: Path | str | None = None,
    panel: str = "6a",
    late_lo: float = 4.0,
    ratio_tol: float = DEFAULT_RATIO_TOL,
    baseline_key: str | None = None,
    qat_trace: tuple[np.ndarray, np.ndarray] | None = None,
    stim_actions: dict[str, list[int]] | None = None,
    open_loop_override: bool = False,
) -> dict[str, Any]:
    """Fig 6a/6b: fp32/PTQ suppressed vs QAT elevated, paper-relative post means.

    Extra paper-faithful checks (when provided):
    - ``stim_actions``: reject a *shared* identical constant lock across
      fp32+PTQ (that greenwash treated distinct constants as Pass while the
      panel is still open-loop). Per-series constant greedy is allowed when
      honest closed-loop (scalar $P_\\beta$ policies often lock).
    - ``open_loop_override``: hard-fail when eval used weak-action / open-loop
      locks instead of the trained / quantized policy.
    - ``qat_trace`` ``(times, y)``: QAT must stay elevated late (no end crash);
      paper QAT [10,12] stays ~high band (digitized peak−end ≈ 36).
    """
    paper = paper or load_refined(paper_path or refined_path(panel))
    alias = {
        "fp32": ("Fully Trained 45Hz", "Fully Trained 30Hz", "fp32", "Fully Trained"),
        "ptq_int8": ("PTQ, INT8", "ptq-int8", "ptq_int8"),
        "ptq_fp16": ("PTQ, FP16", "ptq-fp16", "ptq_fp16"),
        "qat": ("QAT", "qat"),
    }

    def _ours(key: str) -> float:
        for a in alias[key]:
            if a in post_means:
                return float(post_means[a])
        return float("nan")

    def _paper(key: str) -> float:
        for a in alias[key]:
            if a in paper:
                return window_mean(*paper[a], lo=late_lo)
        return float("nan")

    o = {k: _ours(k) for k in alias}
    p = {k: _paper(k) for k in alias}
    base = o.get(baseline_key) if baseline_key else o["fp32"]
    if not np.isfinite(base):
        base = o["fp32"]

    gates: dict[str, bool] = {}
    if np.isfinite(o["fp32"]) and np.isfinite(o["qat"]):
        gates["qat_elevated_vs_fp32"] = o["qat"] > o["fp32"]
    for k in ("fp32", "ptq_int8", "ptq_fp16", "qat"):
        if np.isfinite(o[k]) and np.isfinite(p[k]) and np.isfinite(p["fp32"]):
            gates[f"{k}_level_ratio_near_paper"] = ratio_close(
                o[k], max(abs(base), 1.0), p[k], max(abs(p["fp32"]), 1.0), tol=ratio_tol
            )
    if np.isfinite(o["ptq_fp16"]) and np.isfinite(o["fp32"]):
        gates["ptq_fp16_near_fp32"] = rel_close(o["ptq_fp16"], o["fp32"], tol=0.15)
    if np.isfinite(o["ptq_int8"]) and np.isfinite(o["fp32"]):
        gates["ptq_int8_near_fp32"] = rel_close(o["ptq_int8"], o["fp32"], tol=0.20)

    notes = [
        "Paper QAT/PTQ wiggles are one seed; gates use post-onset means and ratios.",
        "Paper: PTQ tracks fp32 suppression; 10-ep QAT fails to suppress (stays elevated).",
    ]

    if open_loop_override:
        gates["not_open_loop_override"] = False
        notes.append("Eval used open-loop / weak-action lock (not paper closed-loop).")
    else:
        gates["not_open_loop_override"] = True

    if stim_actions:
        # Reject shared identical constant lock across fp32+PTQ (greenwash).
        # Per-series constant greedy under honest closed-loop is allowed —
        # plant dynamics still produce the paper's wiggly suppressed traces.
        def _acts(key: str) -> list[int]:
            for alias_key in (key, key.replace("_", "-"), key.replace("-", "_")):
                if alias_key in stim_actions and stim_actions[alias_key] is not None:
                    return [int(a) for a in stim_actions[alias_key]]
            return []

        fp32 = _acts("fp32")
        ptq16 = _acts("ptq-fp16")
        ptq8 = _acts("ptq-int8")
        shared_constant = bool(
            fp32
            and len(set(fp32)) <= 1
            and ptq16
            and ptq8
            and ptq16 == fp32
            and ptq8 == fp32
        )
        gates["not_shared_constant_action_lock"] = not shared_constant
        notes.append(
            "fp32/PTQ unique actions: "
            f"fp32={len(set(fp32)) if fp32 else 0} "
            f"ptq16={len(set(ptq16)) if ptq16 else 0} "
            f"ptq8={len(set(ptq8)) if ptq8 else 0}; "
            f"shared_constant_lock={shared_constant}"
        )

    if qat_trace is not None:
        qt, qy = qat_trace
        qt = np.asarray(qt, dtype=float)
        qy = np.asarray(qy, dtype=float)
        early_post = window_mean(qt, qy, lo=2.0, hi=8.0)
        late_post = window_mean(qt, qy, lo=10.0, hi=12.0)
        # Prefer a short end band over a single sample — trailing windows are
        # noisy at exact t=12, and max(t≥2) often includes the shared onset
        # baseline (~500) rather than a QAT-only spike.
        end_band = window_mean(qt, qy, lo=11.0, hi=12.0)
        post_mask = (qt >= 3.0) & (qt <= 12.0)
        peak_post = float(np.max(qy[post_mask])) if np.any(post_mask) else float("nan")
        # Paper late QAT stays in the elevated band (digitized ~430–450 at end).
        # Reject late fade into the suppressed band (~fp32 post), not onset noise.
        gates["qat_late_sustained"] = bool(
            np.isfinite(late_post)
            and np.isfinite(early_post)
            and np.isfinite(end_band)
            and late_post >= 0.90 * early_post
            and end_band >= 0.88 * early_post
            and (not np.isfinite(peak_post) or (peak_post - end_band) <= 90.0)
        )
        notes.append(
            f"QAT early_post[2,8]={early_post:.1f} late[10,12]={late_post:.1f} "
            f"end_band[11,12]={end_band:.1f} "
            f"peak_post[3,12]-end_band="
            f"{peak_post - end_band if np.isfinite(peak_post) and np.isfinite(end_band) else float('nan'):.1f}"
        )

    return _gate_pack(
        gates,
        {"ours_post": o, "paper_post": p, "baseline_used": base},
        paper_ref={"path": str(paper_path or refined_path(panel)), "late_lo": late_lo},
        notes=notes,
    )
