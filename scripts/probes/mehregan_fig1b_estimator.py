"""Mehregan Fig 1b — does the PSD estimator choice move the PD peak to ~12.6 Hz?

Diagnostic probe (not a ship surface). Simulates the three Fig 1b conditions
(healthy, PD, PD + 130 Hz STN DBS) with the panel's own ``simulate_condition``
from ``scripts/figures/papers/mehregan/1b/plot.py``, caches the GPi spike trains,
then recomputes the neuron-averaged PSD under several estimator variants and
scores each against the refined paper digitization.

Run (single thread, from the worktree; the run is wrapped by the caller):
  timeout 1800 nice -n 10 uv run --no-sync python -m rl_adaptive_dbs.run \
      --max-threads 1 scripts/probes/mehregan_fig1b_estimator.py

Outputs: a compact table on stdout and JSON at ``--out`` (default
``~/research/rl-adaptive-dbs/artifacts/probes/mehregan_fig1b_estimator.json``).
Spike cache: ``/tmp/mehregan_fig1b_spikes.npz`` (reused when it matches seeds and duration).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch

from rl_adaptive_dbs.panel import load_script_module

from envs.plant import PythonPlant
from envs.plant.biomarkers import SpectrumParams, multitaper_psd_point_process

REPO = Path(__file__).resolve().parents[2]
PLOT = REPO / "scripts/figures/papers/mehregan/1b/plot.py"
PAPER_DEFAULT = REPO / "artifacts/figures/papers/mehregan/1b/paper_digitization/curves_wpd_refined.json"
CACHE = Path("/tmp/mehregan_fig1b_spikes.npz")
DEFAULT_OUT = Path.home() / "research/rl-adaptive-dbs/artifacts/probes/mehregan_fig1b_estimator.json"

_PLOT = load_script_module("mehregan_fig1b_plot", PLOT)

DT_MS = 0.01
DUR_S = 10.0
FG = np.arange(1.0, 50.0 + 1e-9, 0.1)  # common 1-50 Hz comparison grid
PAPER_KEYS = {"healthy": "healthy", "pd": "pd", "pd_130hz": "pd_130hz"}
BETA = (13.0, 35.0)


# ---------------------------------------------------------------- simulation

def simulate_or_load(seeds: tuple[int, ...], duration_s: float, cache: Path) -> dict[str, list[list[np.ndarray]]]:
    """Return ``spikes[cond][seed_index] -> list of per-neuron spike-time arrays (s)``."""
    conds = [c.key for c in _PLOT.CONDITIONS]
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta["seeds"] == list(seeds) and meta["duration_s"] == duration_s:
            print(f"reusing spike cache {cache}", file=sys.stderr)
            return {
                c: [
                    [np.asarray(data[f"{c}__{s}__{i}"]) for i in range(meta["n_neurons"][c][str(s)])]
                    for s in seeds
                ]
                for c in conds
            }
        print("spike cache seeds/duration differ; re-simulating", file=sys.stderr)

    captured: list[list[np.ndarray]] = []
    original = _PLOT.mean_gpi_psd

    def spy(gpi_spikes: list[np.ndarray], *, dt_ms: float, segment_duration_s: float):
        captured.append([np.asarray(s, dtype=float).copy() for s in gpi_spikes])
        return original(gpi_spikes, dt_ms=dt_ms, segment_duration_s=segment_duration_s)

    t0 = time.monotonic()
    _PLOT.mean_gpi_psd = spy  # the panel's simulate_condition looks this up at call time
    try:
        with PythonPlant() as plant:
            for cond in _PLOT.CONDITIONS:
                _PLOT.simulate_condition(plant, cond, seeds=seeds, duration_s=duration_s)
    finally:
        _PLOT.mean_gpi_psd = original
    sim_s = time.monotonic() - t0
    print(f"simulated {len(captured)} runs in {sim_s:.1f} s", file=sys.stderr)

    spikes: dict[str, list[list[np.ndarray]]] = {}
    arrays: dict[str, np.ndarray] = {}
    n_neurons: dict[str, dict[str, int]] = {}
    k = 0
    for c in conds:
        spikes[c] = []
        n_neurons[c] = {}
        for s in seeds:
            neurons = captured[k]
            k += 1
            spikes[c].append(neurons)
            n_neurons[c][str(s)] = len(neurons)
            for i, arr in enumerate(neurons):
                arrays[f"{c}__{s}__{i}"] = arr
    meta = {"seeds": list(seeds), "duration_s": duration_s, "n_neurons": n_neurons, "sim_s": sim_s}
    arrays["meta"] = np.array(json.dumps(meta))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    print(f"wrote spike cache {cache}", file=sys.stderr)
    return spikes


# ---------------------------------------------------------------- estimators

def _grid(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.interp(FG, np.asarray(f, dtype=float), np.asarray(p, dtype=float))


def est_panel(neurons: list[np.ndarray]) -> np.ndarray:
    """(a) The panel's own estimator: multitaper TW=3, K=5, 10 s, mean over neurons."""
    f, p = _PLOT.mean_gpi_psd(neurons, dt_ms=DT_MS, segment_duration_s=DUR_S)
    return _grid(f, p)


def est_multitaper(neurons: list[np.ndarray], tw: float, k: int, seg: float = DUR_S) -> np.ndarray:
    params = SpectrumParams(dt_ms=DT_MS, time_bandwidth=tw, n_tapers=k)
    out = []
    for s in neurons:
        p, f = multitaper_psd_point_process(s, params, segment_duration_s=seg)
        out.append(_grid(f, p))
    return np.mean(out, axis=0)


def est_welch_binned(neurons: list[np.ndarray], seg_s: float) -> np.ndarray:
    """(c) 1 ms binned, mean-subtracted spike counts; Hann Welch, 50% overlap."""
    edges = np.arange(0.0, DUR_S + 1e-9, 0.001)
    nperseg = int(round(seg_s * 1000))
    out = []
    for s in neurons:
        counts = np.histogram(s, bins=edges)[0].astype(float)
        counts -= counts.mean()
        f, p = welch(
            counts,
            fs=1000.0,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
        out.append(_grid(f, p))
    return np.mean(out, axis=0)


def est_pop_sum(neurons: list[np.ndarray]) -> np.ndarray:
    """(d) One multitaper PSD (panel settings) of the merged population spike train."""
    merged = np.sort(np.concatenate([np.asarray(s, dtype=float) for s in neurons]))
    params = SpectrumParams(dt_ms=DT_MS)
    p, f = multitaper_psd_point_process(merged, params, segment_duration_s=DUR_S)
    return _grid(f, p)


def est_panel_windows(neurons: list[np.ndarray], win_s: float = 2.0) -> np.ndarray:
    """(e) Panel estimator on consecutive 2 s windows (spikes shifted to t=0), averaged."""
    params = SpectrumParams(dt_ms=DT_MS)
    out = []
    for s in neurons:
        s = np.asarray(s, dtype=float)
        for w in range(int(round(DUR_S / win_s))):
            lo, hi = w * win_s, (w + 1) * win_s
            sel = s[(s >= lo) & (s < hi)] - lo
            p, f = multitaper_psd_point_process(sel, params, segment_duration_s=win_s)
            out.append(_grid(f, p))
    return np.mean(out, axis=0)


VARIANTS: dict[str, Any] = {
    "a_panel_mt_TW3K5_10s": est_panel,
    "b1_mt_TW1.5K2_10s": lambda n: est_multitaper(n, 1.5, 2),
    "b2_mt_TW5K9_10s": lambda n: est_multitaper(n, 5.0, 9),
    "c1_welch_binned_1s": lambda n: est_welch_binned(n, 1.0),
    "c2_welch_binned_2s": lambda n: est_welch_binned(n, 2.0),
    "d_pop_sum_mt_10s": est_pop_sum,
    "e_panel_mt_2s_windows": est_panel_windows,
}


# ---------------------------------------------------------------- scoring

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / den) if den > 0 else float("nan")


def unit_max(y: np.ndarray) -> np.ndarray:
    m = float(np.max(y))
    return y / m if m > 0 else y * np.nan


def peak_hz(y: np.ndarray) -> float:
    return float(FG[int(np.argmax(y))])


def beta_frac(y: np.ndarray) -> float:
    m = (FG >= BETA[0]) & (FG <= BETA[1])
    return float(np.mean(unit_max(y)[m]))


def load_paper(path: Path) -> dict[str, np.ndarray]:
    series = json.loads(path.read_text())["series"]
    out = {}
    for key in PAPER_KEYS:
        xy = series[key]["xy"]
        x = np.asarray(xy["x"], dtype=float)
        y = np.asarray(xy["y"], dtype=float)
        order = np.argsort(x, kind="stable")
        out[key] = np.interp(FG, x[order], y[order])
    return out


def score(psd_seed: np.ndarray, paper: np.ndarray) -> dict[str, float]:
    """psd_seed: 1-D PSD on FG. Pearson on unit-max normalised curves (== raw Pearson)."""
    return {
        "peak_hz": peak_hz(psd_seed),
        "r_norm": pearson(unit_max(psd_seed), unit_max(paper)),
        "r_raw": pearson(psd_seed, paper),
        "beta_frac": beta_frac(psd_seed),
    }


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--duration-s", type=float, default=DUR_S)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--paper", type=Path, default=PAPER_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    seeds = _PLOT.parse_seeds(args.seeds)
    t_all = time.monotonic()
    spikes = simulate_or_load(seeds, args.duration_s, args.cache)
    paper = load_paper(args.paper)

    results: dict[str, Any] = {}
    for vname, fn in VARIANTS.items():
        t0 = time.monotonic()
        per_cond: dict[str, Any] = {}
        for c in _PLOT.CONDITIONS:
            key = c.key
            psds = np.vstack([fn(neurons) for neurons in spikes[key]])  # (n_seeds, FG)
            mean_psd = psds.mean(axis=0)
            s0 = score(psds[0], paper[key])
            m5 = score(mean_psd, paper[key])
            per_seed = [score(p, paper[key]) for p in psds]
            per_cond[key] = {
                "seed0": s0,
                "mean": m5,
                "per_seed_peak_hz": [d["peak_hz"] for d in per_seed],
                "per_seed_r_norm": [d["r_norm"] for d in per_seed],
                "mean_psd_unit_max": unit_max(mean_psd).round(6).tolist(),
            }
        results[vname] = {"per_condition": per_cond, "elapsed_s": round(time.monotonic() - t0, 2)}
        print(f"  variant {vname} done in {results[vname]['elapsed_s']:.1f} s", file=sys.stderr, flush=True)

    paper_peaks = {k: {"peak_hz": peak_hz(paper[k]), "beta_frac": beta_frac(paper[k])} for k in PAPER_KEYS}
    payload = {
        "figure": "mehregan_fig1b_estimator_probe",
        "seeds": list(seeds),
        "duration_s": args.duration_s,
        "grid_hz": [float(FG[0]), float(FG[-1]), float(FG[1] - FG[0])],
        "paper_digitization": str(args.paper),
        "paper": paper_peaks,
        "variants": results,
        "runtime_s": round(time.monotonic() - t_all, 1),
        "note": "Pearson r is invariant to scale, so r_norm == r_raw by construction.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    # Compact table
    hdr = (
        f"{'variant':24s} | {'PD peak s0/m5':>14s} | {'PD r s0/m5':>13s} | "
        f"{'130 peak s0/m5':>15s} | {'130 r s0/m5':>13s} | {'PD beta% m5':>11s} | {'HC peak m5':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for vname, res in results.items():
        pc = res["per_condition"]
        pd, p130, hc = pc["pd"], pc["pd_130hz"], pc["healthy"]
        print(
            f"{vname:24s} | {pd['seed0']['peak_hz']:6.1f}/{pd['mean']['peak_hz']:5.1f} | "
            f"{pd['seed0']['r_norm']:+6.2f}/{pd['mean']['r_norm']:+5.2f} | "
            f"{p130['seed0']['peak_hz']:7.1f}/{p130['mean']['peak_hz']:5.1f} | "
            f"{p130['seed0']['r_norm']:+6.2f}/{p130['mean']['r_norm']:+5.2f} | "
            f"{100 * pd['mean']['beta_frac']:11.1f} | {hc['mean']['peak_hz']:10.1f}"
        )
    print(
        f"paper: PD peak {paper_peaks['pd']['peak_hz']:.1f} Hz, 130 Hz peak "
        f"{paper_peaks['pd_130hz']['peak_hz']:.1f} Hz, healthy {paper_peaks['healthy']['peak_hz']:.1f} Hz; "
        f"PD beta% {100 * paper_peaks['pd']['beta_frac']:.1f}"
    )
    within = [
        v for v, r in results.items()
        if abs(r["per_condition"]["pd"]["mean"]["peak_hz"] - 12.6) <= 3.0
        or abs(r["per_condition"]["pd"]["seed0"]["peak_hz"] - 12.6) <= 3.0
    ]
    print(f"variants with PD peak within +/-3 Hz of 12.6 Hz (seed0 or mean): {within or 'none'}")
    print(f"wrote {args.out}; total runtime {payload['runtime_s']:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
