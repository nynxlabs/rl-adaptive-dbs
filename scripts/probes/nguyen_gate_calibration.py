"""Calibrate the Nguyen Fig 4–7 gate tolerances against seed-to-seed noise.

Noise model (docs/controllers/snn/figures/4.md § Frozen DSQN layers, gate calibration): for a
config with N finished seeds, each seed's per-episode series minus the config's mean
series, inflated by √(N/(N−1)), is one residual. A synthetic *faithful run* is the
digitized paper curve on the episode grid plus one residual:

- absolute residuals (same units as the paper): episode length, reward (our Eq. (7)
  scale matches the paper's ep 0–50 level), α–β (same units and θ);
- relative residuals ``seed / config mean`` multiplied onto the paper curve: DBS
  energy, frequency and spike count (our levels / count definition differ from the
  paper's, so only the proportional spread transfers);
- late-CV checks measure the noise itself, so for them the residuals are rescaled to
  the paper's late spread (``paper CV × paper late mean``) before adding them.

For every check the script prints the paper value, the implemented tolerance, the
smallest tolerance at which ≥ ``--min-pass`` of synthetic runs pass, the synthetic pass
rate, and the values on the official run and on each pool seed, then the joint pass
rate of a synthetic faithful run across all required gates of each panel.

Fig 7 uses the paper's band instead of training seeds: each synthetic eval is the
paper average plus 50 episode residuals bootstrapped from our 50-episode eval
(``7/eval.json``), rescaled per step to the paper's single-episode SD.

Usage::

    uv run --no-sync python scripts/probes/nguyen_gate_calibration.py [--json-out PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "digitization"))

import nguyen_gates as ng  # noqa: E402

NGUYEN = Path("artifacts/figures/papers/nguyen")
CANDIDATES = NGUYEN / "4" / "candidates"
OFFICIAL_SERIES = NGUYEN / "4" / "series.json"
OFFICIAL_EVAL = NGUYEN / "7" / "eval.json"
POOL_PATTERNS = ("wu1500-seed*", "s5d1-seed*", "audit1-base-s*", "audit1-net1*-s*")
N = ng.FIG4_EPISODES
SPIKE_FIELDS = ("episode_spikes_per_step", "episode_spike_totals")
FIG7_BOOTSTRAPS = 400


# ---------------------------------------------------------------------------
# Pools and residuals
# ---------------------------------------------------------------------------


def config_name(run_dir: str) -> str:
    return re.sub(r"-(seed|s)\d+$", "", Path(run_dir).name)


def load_pools() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    pools: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for pattern in POOL_PATTERNS:
        for run_dir in sorted(glob.glob(str(CANDIDATES / pattern))):
            path = Path(run_dir) / "series.json"
            if not path.exists():
                continue
            series = json.loads(path.read_text(encoding="utf-8"))
            if len(series.get("episode_lengths", [])) < N:
                continue
            pools[config_name(run_dir)].append((Path(run_dir).name, series))
    return {k: v for k, v in pools.items() if len(v) >= 2}


def spike_field(series: dict[str, Any]) -> str | None:
    return next((f for f in SPIKE_FIELDS if f in series), None)


def residuals(
    runs: list[tuple[str, dict[str, Any]]], field: str, *, relative: bool
) -> list[tuple[str, np.ndarray]]:
    """Per-seed residuals of ``field`` vs the config mean, inflated by √(N/(N−1))."""
    arr = np.asarray([np.asarray(s[field][:N], dtype=float) for _, s in runs])
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    k = math.sqrt(n / (n - 1))
    if relative:
        res = (arr / np.where(np.abs(mean) > 1e-12, mean, np.nan) - 1.0) * k
        res = np.nan_to_num(res)
    else:
        res = (arr - mean) * k
    return [(name, res[i]) for i, (name, _) in enumerate(runs)]


def paper_grids() -> dict[str, np.ndarray]:
    ref = ng.fig4_paper_reference()
    return {
        "length": ref["length_grid"],
        "reward": ref["reward_grid"],
        # Spike Count is a raw per-episode trace: smooth it so noise is not counted twice.
        "spikes": ng.centered_smooth(ng._paper_grid("fig5_spikes", "Spike Count"), ng.FIG4_SMOOTH),
        "energy": ng._paper_grid("fig5_energy", "Smoothed"),
        "ab": ng._paper_grid("fig6_power", "Smoothed"),  # flat before the first traced episode
        "freq": ng._paper_grid("fig6_freq", "Smoothed"),
        "amp": ng._paper_grid("fig6_amp", "Smoothed"),
        "pw": ng._paper_grid("fig6_pw", "Smoothed"),
    }


def synthetic_runs(pools: dict[str, list[tuple[str, dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    """Synthetic faithful runs per panel (each a dict of per-episode series)."""
    g = paper_grids()
    paper_m6 = ng.fig6_paper_metrics()
    late = slice(ng.FIG6_LATE_LO, N)
    fig4: list[dict[str, Any]] = []
    fig5: list[dict[str, Any]] = []
    fig6_main: list[dict[str, Any]] = []
    fig6_cv: list[dict[str, Any]] = []
    for cfg, runs in pools.items():
        r_len = residuals(runs, "episode_lengths", relative=False)
        r_rew = residuals(runs, "episode_rewards", relative=False)
        for (name, rl), (_, rr) in zip(r_len, r_rew):
            lengths = np.clip(np.rint(g["length"] + rl), 1.0, 25.0)
            fig4.append({"name": name, "cfg": cfg, "rewards": g["reward"] + rr, "lengths": lengths})
        sf = spike_field(runs[0][1])
        if sf and all(sf in s for _, s in runs):
            r_sp = residuals(runs, sf, relative=True)
            r_en = residuals(runs, "episode_energies", relative=True)
            for (name, rs), (_, re_) in zip(r_sp, r_en):
                fig5.append({
                    "name": name, "cfg": cfg,
                    "spikes": g["spikes"] * (1.0 + rs), "energies": g["energy"] * (1.0 + re_),
                })
        if all("episode_alpha_beta_finals" in s for _, s in runs):
            r_ab = residuals(runs, "episode_alpha_beta_finals", relative=False)
            r_fr = residuals(runs, "episode_frequencies", relative=True)
            r_am = residuals(runs, "episode_amplitudes", relative=True)
            r_pw = residuals(runs, "episode_pulse_widths", relative=True)
            # Late-CV copies: rescale to the paper's late spread (pooled factor per config).
            sd_ab = float(np.sqrt(np.mean([np.var(r[late]) for _, r in r_ab])))
            sd_fr = float(np.sqrt(np.mean([np.var(r[late]) for _, r in r_fr])))
            k_ab = paper_m6["ab_late_cv"] * float(np.mean(g["ab"][late])) / max(sd_ab, 1e-12)
            k_fr = paper_m6["freq_late_cv"] / max(sd_fr, 1e-12)
            for (name, ra), (_, rf), (_, ram), (_, rpw) in zip(r_ab, r_fr, r_am, r_pw):
                base = {"name": name, "cfg": cfg, "amp": g["amp"] * (1.0 + ram), "pw": g["pw"] * (1.0 + rpw)}
                fig6_main.append({**base, "ab": g["ab"] + ra, "freq": g["freq"] * (1.0 + rf)})
                fig6_cv.append({**base, "ab": g["ab"] + k_ab * ra, "freq": g["freq"] * (1.0 + k_fr * rf)})
    return {"4": fig4, "5": fig5, "6": fig6_main, "6cv": fig6_cv}


def fig7_synthetic(rng: np.random.Generator) -> list[list[list[float]]]:
    payload = json.loads(OFFICIAL_EVAL.read_text(encoding="utf-8"))
    ref = ng._fig7_paper_cached()
    steps = ng.FIG7_STEPS
    trajs = np.asarray([tr[:steps] for tr in payload["alpha_beta_trajectories"] if len(tr) >= steps], dtype=float)
    n = trajs.shape[0]
    our_sd = trajs.std(axis=0, ddof=1)
    res = (trajs - trajs.mean(axis=0)) * (ref["sd"] / np.maximum(our_sd, 1e-9))
    out = []
    for _ in range(FIG7_BOOTSTRAPS):
        pick = rng.integers(0, n, size=ng.PAPER_FIG7_EPISODES)
        out.append((ref["avg"] + res[pick]).tolist())
    return out


# ---------------------------------------------------------------------------
# Gate evaluation per panel
# ---------------------------------------------------------------------------


def eval_fig4(run: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    out = ng.fig4_gates(run["rewards"], run["lengths"], max_episode_steps=25)
    gates = {k: out[g][k] for g, keys in ng.FIG4_REQUIRED_KEYS.items() for k in keys}
    metrics = {f"{g}.{k}": v for g in ("reward", "length") for k, v in out[g]["metrics"].items()}
    return gates, metrics


def eval_fig5(run: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    out = ng.fig5_spikes_energy_gates(run["spikes"], run["energies"])
    return dict(out["gates"]), out["metrics"]


def eval_fig6(run: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    out = ng.fig6_training_gates(run["ab"], run["amp"], run["freq"], run["pw"])
    return dict(out["gates"]), out["metrics"]


def eval_fig7(trajs: list[list[float]]) -> tuple[dict[str, bool], dict[str, Any]]:
    out = ng.fig7_eval_gates(trajs)
    m = dict(out["metrics"])
    m["start"], m["peak"] = m.get("start_val"), m.get("peak_val")
    return dict(out["gates"]), m


PANELS: dict[str, dict[str, Any]] = {
    "4": {"checks": lambda: ng.FIG4_CHECKS, "paper": ng.fig4_paper_metrics, "eval": eval_fig4,
          "required": lambda: [k for keys in ng.FIG4_REQUIRED_KEYS.values() for k in keys]},
    "5": {"checks": lambda: ng.FIG5_CHECKS, "paper": ng.fig5_paper_metrics, "eval": eval_fig5,
          "required": lambda: list(ng.FIG5_REQUIRED_KEYS)},
    "6": {"checks": lambda: ng.FIG6_CHECKS, "paper": ng.fig6_paper_metrics, "eval": eval_fig6,
          "required": lambda: list(ng.FIG6_REQUIRED_KEYS)},
    "7": {"checks": ng.fig7_checks, "paper": ng.fig7_paper_metrics, "eval": eval_fig7,
          "required": lambda: [
              "eval_protocol_ok", "step_series_finite", "peak_step_timing",
              *dict.fromkeys(c.gate for c in ng.fig7_checks()), "pearson_shape_ok",
          ]},
}
CV_METRICS = frozenset({"ab_late_cv", "freq_late_cv"})


def needed_tol(devs: list[float], min_pass: float) -> float:
    """Smallest tolerance at which ≥ ``min_pass`` of the deviations pass."""
    d = sorted(devs)
    k = max(1, math.ceil(min_pass * len(d)))
    return d[k - 1]


TIMING_METRICS = ("t10_lead", "t50", "t90")


def clean_tol(needed: float, metric: str) -> float:
    """Round outward: episode milestones to a multiple of 5, everything else to 2 significant figures."""
    if not math.isfinite(needed):
        return float("inf")
    if needed <= 0:
        return 0.0
    if metric.endswith(TIMING_METRICS):
        return 5.0 * math.ceil(needed / 5.0 - 1e-9)
    mag = 10 ** (math.floor(math.log10(needed)) - 1)
    return math.ceil(needed / mag - 1e-9) * mag


def key_level_tolerances(
    checks: tuple[Any, ...], devs: dict[int, list[float]], min_pass: float
) -> dict[str, float]:
    """Tolerances for gate keys made of several checks.

    Each check starts at its own clean 90 % tolerance; candidates are the clean-rounded
    synthetic deviations above it. Pick the combination with the smallest widening
    ``max_j(tol_j / own_j)`` at which the key as a whole passes on ≥ ``min_pass`` of runs.
    """
    import itertools

    out: dict[str, float] = {}
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(checks):
        by_key[c.gate].append(i)
    for idx in by_key.values():
        if len(idx) < 2:
            continue
        mat = np.asarray([devs[i] for i in idx])  # checks × runs
        own = [clean_tol(needed_tol(list(mat[j]), min_pass), checks[i].metric) for j, i in enumerate(idx)]
        cands = []
        for j, i in enumerate(idx):
            vals = {own[j]} | {
                clean_tol(float(d), checks[i].metric) for d in mat[j] if math.isfinite(d) and d > own[j]
            }
            cands.append(sorted(vals))
        best: tuple[float, tuple[float, ...]] | None = None
        for combo in itertools.product(*cands):
            tols = np.asarray(combo)[:, None]
            if float(np.mean(np.all(mat <= tols, axis=0))) < min_pass:
                continue
            widen = max(t / o if o > 0 else (1.0 if t == 0 else math.inf) for t, o in zip(combo, own))
            if best is None or widen < best[0]:
                best = (widen, combo)
        chosen = best[1] if best else tuple(math.inf for _ in idx)
        for j, i in enumerate(idx):
            out[checks[i].metric] = chosen[j]
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return "T" if v else "F"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if not math.isfinite(v):
        return "nan"
    a = abs(v)
    if a >= 1e4:
        return f"{v:.3g}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def calibrate_panel(
    panel: str,
    synth: list[Any],
    synth_cv: list[Any] | None,
    real: list[tuple[str, Any]],
    *,
    min_pass: float,
) -> dict[str, Any]:
    spec = PANELS[panel]
    checks = spec["checks"]()
    paper = spec["paper"]()
    required = spec["required"]()
    ev = spec["eval"]
    s_out = [ev(r) for r in synth]
    s_cv_out = [ev(r) for r in synth_cv] if synth_cv else None
    r_out = [(name, ev(r)) for name, r in real]

    rows = []
    devs_by_check: dict[int, list[float]] = {}
    for i, c in enumerate(checks):
        src = s_cv_out if (c.metric in CV_METRICS and s_cv_out) else s_out
        devs_by_check[i] = [ng.check_deviation(c, m.get(c.metric), paper.get(c.metric)) for _, m in src]
    key_suggest = key_level_tolerances(checks, devs_by_check, min_pass)
    for c in checks:
        src = s_cv_out if (c.metric in CV_METRICS and s_cv_out) else s_out
        devs = [ng.check_deviation(c, m.get(c.metric), paper.get(c.metric)) for _, m in src]
        need = needed_tol(devs, min_pass)
        rate = float(np.mean([d <= c.tol for d in devs]))
        real_vals = []
        for name, (_, m) in r_out:
            v = m.get(c.metric)
            real_vals.append((name, v, ng.check_deviation(c, v, paper.get(c.metric)) <= c.tol))
        rows.append({
            "gate": c.gate, "metric": c.metric, "kind": c.kind, "note": c.note,
            "paper": paper.get(c.metric), "tol": c.tol, "needed_tol": need,
            "suggested_tol": key_suggest.get(c.metric, clean_tol(need, c.metric)),
            "synthetic_pass_rate": rate, "real": real_vals,
            "noise": "paper-spread rescaled" if (c.metric in CV_METRICS and s_cv_out) else "seed residuals",
        })

    # Joint pass over all required gates (CV gates read from the rescaled copy of the same seed).
    cv_gates = {c.gate for c in checks if c.metric in CV_METRICS}
    joint = []
    gate_rates: dict[str, float] = {}
    for key in required:
        vals = []
        for i, (g, _) in enumerate(s_out):
            src = s_cv_out[i][0] if (key in cv_gates and s_cv_out) else g
            vals.append(bool(src.get(key, False)))
        gate_rates[key] = float(np.mean(vals))
    for i, (g, _) in enumerate(s_out):
        ok = True
        for key in required:
            src = s_cv_out[i][0] if (key in cv_gates and s_cv_out) else g
            ok = ok and bool(src.get(key, False))
        joint.append(ok)
    real_pass = {name: {k: bool(g.get(k, False)) for k in required} for name, (g, _) in r_out}
    return {
        "panel": panel, "n_synthetic": len(s_out), "rows": rows, "gate_rates": gate_rates,
        "joint_pass_rate": float(np.mean(joint)) if joint else float("nan"),
        "real_pass": real_pass, "required": required,
    }


def print_panel(res: dict[str, Any], *, verbose: bool) -> None:
    print(f"\n=== Fig {res['panel']} — {res['n_synthetic']} synthetic faithful runs ===")
    print(f"{'check (gate / metric)':58s} {'kind':4s} {'paper':>8s} {'tol':>7s} {'needed':>7s} {'suggest':>7s} {'synth':>6s}  official")
    for r in res["rows"]:
        off = r["real"][0] if r["real"] else ("—", None, False)
        label = f"{r['gate']} / {r['metric']}"
        print(
            f"{label[:58]:58s} {r['kind']:4s} {fmt(r['paper']):>8s} {fmt(r['tol']):>7s} "
            f"{fmt(r['needed_tol']):>7s} {fmt(r['suggested_tol']):>7s} {r['synthetic_pass_rate']:6.0%}  "
            f"{fmt(off[1])} {'pass' if off[2] else 'FAIL'}"
        )
        if verbose and len(r["real"]) > 1:
            vals = " ".join(f"{fmt(v)}{'' if ok else '*'}" for _, v, ok in r["real"][1:])
            print(f"    pool seeds (* = fail): {vals}")
    print("required gate → synthetic pass rate:")
    for k, v in res["gate_rates"].items():
        print(f"    {k:44s} {v:6.0%}")
    print(f"JOINT synthetic pass rate (all required gates): {res['joint_pass_rate']:.0%}")
    names = list(res["real_pass"])
    if names:
        print("real runs, failed required gates:")
        for name in names:
            failed = [k for k, ok in res["real_pass"][name].items() if not ok]
            print(f"    {name:24s} {'PASS' if not failed else 'fail: ' + ', '.join(failed)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--min-pass", type=float, default=0.90)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--panels", default="4,5,6,7")
    ap.add_argument("--quiet", action="store_true", help="omit per-seed value lists")
    args = ap.parse_args(argv)

    pools = load_pools()
    print("residual pools:", {k: len(v) for k, v in pools.items()})
    synth = synthetic_runs(pools)
    official = json.loads(OFFICIAL_SERIES.read_text(encoding="utf-8"))
    from rl_adaptive_dbs.panel import load_script_module

    fig5_plot = load_script_module("nguyen_fig5_plot", ROOT / "scripts/figures/papers/nguyen/5/plot.py")

    def real_runs(panel: str) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        if panel == "4":
            out.append(("official", {"rewards": official["episode_rewards"], "lengths": official["episode_lengths"]}))
        elif panel == "5":
            out.append(("official", {"spikes": fig5_plot.network_spikes(official), "energies": official["episode_energies"]}))
        elif panel == "6":
            out.append(("official", {
                "ab": official["episode_alpha_beta_finals"], "freq": official["episode_frequencies"],
                "amp": official["episode_amplitudes"], "pw": official["episode_pulse_widths"],
            }))
        elif panel == "7":
            return [("official", json.loads(OFFICIAL_EVAL.read_text(encoding="utf-8"))["alpha_beta_trajectories"])]
        for runs in pools.values():
            for name, s in runs:
                if panel == "4":
                    out.append((name, {"rewards": s["episode_rewards"], "lengths": s["episode_lengths"]}))
                elif panel == "5" and spike_field(s):
                    # Pool runs lack per-population counts: total spikes stand in (scale-free rows only).
                    out.append((name, {"spikes": s[spike_field(s)], "energies": s["episode_energies"]}))
                elif panel == "6" and "episode_alpha_beta_finals" in s:
                    out.append((name, {
                        "ab": s["episode_alpha_beta_finals"], "freq": s["episode_frequencies"],
                        "amp": s["episode_amplitudes"], "pw": s["episode_pulse_widths"],
                    }))
        return out

    results = []
    for panel in args.panels.split(","):
        if panel == "7":
            synth_runs: list[Any] = fig7_synthetic(np.random.default_rng(0))
            synth_cv = None
        else:
            synth_runs = synth[panel]
            synth_cv = synth.get("6cv") if panel == "6" else None
        res = calibrate_panel(panel, synth_runs, synth_cv, real_runs(panel), min_pass=args.min_pass)
        print_panel(res, verbose=not args.quiet)
        results.append(res)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
