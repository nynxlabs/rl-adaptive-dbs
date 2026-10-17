"""Mehregan Fig 1b — which plant parameter sets the PD GPi beta peak frequency?

Diagnostic probe (not a ship surface). Runs one PD (or healthy) condition with
one-factor-at-a-time parameter changes applied by patching the plant from the
outside (the default plant code is untouched):

* synaptic kernels (``integrator._synaptic_kernels``): amplitude scale ``amp_<k>``
  (equivalent to scaling that synapse's conductance) and time stretch ``ts_<k>``
  (delay and rise/decay times all scaled together), ``dl_<k>`` extra delay in ms, for ``k`` in the
  ``build_synaptic_kernels`` keys without the ``syn_func_`` prefix;
* network draws (``NetworkInitDraws.draw``): ``scale_<field>`` for gsngea,
  gsngen, gsngi, ggege, gcorsna, gcorsnn, gcordrstr;
* loop scalars (``integrator.run_cbgt_loop`` positional args): ``iappgpe`` and
  ``pd_eff`` (the lesion factor fed to Im and GPe-GPe in the loop, as a float).
  ``gm_pd`` sets the striatal M-current factor ``2.6 - 1.1*pd`` alone (GPe-GPe
  held at its pd=1 strength by rescaling ggege).

Reports the GPi seed-mean PSD peak (1-50 Hz, gate smoothing), the 13-35 Hz
window mean, the 1-12 Hz window mean, and STN / GPe / Str-D2 peaks and rates.

Run (single thread, from the worktree; the caller wraps it):
  timeout 900 heavy --no-queue nice -n 10 uv run --no-sync python -m rl_adaptive_dbs.run \
      --max-threads 1 scripts/probes/mehregan_fig1b_plant_beta.py --set amp_gpe_stn=0.5
Appends one JSON line per run to ``--out``.

Findings (Oct 17 2026, PD seed 0, 10 s; 30 runs): the PD GPi peak is a
GPe-STN(-GPi) rhythm that persists with cortex and striatum cut (16.5 Hz) and
stays at 16.5-19.3 Hz under every one-factor change that keeps it — GPe-GPe
x0.25/x2, striatal Im factor 1.2/2.0, lesion factor 0.5, iappgpe 1/2, GPe->STN
x0.5/x2, STN->GPe x0.5/x2, GPe->STN delay +4/+8 ms, loop kernels stretched
x1.5. Changes that remove it switch regime instead (healthy-like spectrum, or
a 7.5-8.6 Hz STN mode at GPe->STN stretch x2); none lands near 12.6 Hz.
Seeds with MATLAB-exported draws (tests/fixtures) carry pd=1 gcordrstr into
pd=0 runs; correcting it for healthy seed 0 changes nothing (D1 is silent).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.biomarkers import SpectrumParams, multitaper_psd_point_process
from envs.plant import python_backend as pb
from envs.plant.network import integrator as integ

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("/tmp/meh1b-plant/plant_beta_probe.jsonl")
DRAW_FIELDS = ("gsngea", "gsngen", "gsngi", "ggege", "gcorsna", "gcorsnn", "gcordrstr")


def _stretch(kernel: np.ndarray, dt: float, s: float) -> np.ndarray:
    t = np.arange(kernel.size) * dt
    return np.interp(t / s, t, kernel, left=0.0, right=0.0)


def install_patches(knobs: dict[str, float], dt_ms: float) -> None:
    kern_amp = {k[4:]: v for k, v in knobs.items() if k.startswith("amp_")}
    kern_ts = {k[3:]: v for k, v in knobs.items() if k.startswith("ts_")}
    kern_dl = {k[3:]: v for k, v in knobs.items() if k.startswith("dl_")}  # extra delay, ms
    draw_scale = {k[6:]: v for k, v in knobs.items() if k.startswith("scale_")}
    for name in draw_scale:
        if name not in DRAW_FIELDS:
            raise SystemExit(f"unknown draw field {name}")
    pd_eff = knobs.get("pd_eff")
    if "gm_pd" in knobs:
        pd_eff = (2.6 - knobs["gm_pd"]) / 1.1
        # hold GPe-GPe at the pd=1 multiplier (1.0) despite the changed loop pd
        draw_scale["ggege"] = draw_scale.get("ggege", 1.0) / (0.25 * (3.0 * pd_eff + 1.0))

    orig_k = integ.build_synaptic_kernels
    base = orig_k(dt_ms)
    for k in (*kern_amp, *kern_ts, *kern_dl):
        if f"syn_func_{k}" not in base:
            raise SystemExit(f"unknown kernel {k}")

    def kernels(dt: float) -> dict[str, np.ndarray]:
        out = dict(orig_k(dt))
        for k, s in kern_ts.items():
            out[f"syn_func_{k}"] = _stretch(out[f"syn_func_{k}"], dt, s)
        for k, d in kern_dl.items():
            kk = out[f"syn_func_{k}"]
            t = np.arange(kk.size) * dt
            out[f"syn_func_{k}"] = np.interp(t - d, t, kk, left=0.0, right=0.0)
        for k, a in kern_amp.items():
            out[f"syn_func_{k}"] = out[f"syn_func_{k}"] * a
        return out

    integ._synaptic_kernels = kernels  # type: ignore[assignment]

    gcordrstr_base = knobs.get("gcordrstr_base")  # replaces the 0.07-0.044*pd mean

    def _apply(d, pd_draw):  # noqa: ANN001, ANN202
        if d is None:
            return d
        upd = {f: getattr(d, f) * draw_scale[f] for f in draw_scale}
        if gcordrstr_base is not None:
            jitter = d.gcordrstr - (0.07 - 0.044 * pd_draw)  # the 0.001*rand part
            upd["gcordrstr"] = gcordrstr_base + jitter * (draw_scale.get("gcordrstr", 1.0))
        return replace(d, **upd) if upd else d

    orig_draw = integ.NetworkInitDraws.draw.__func__

    def draw(cls, rng, *, n, pd):  # noqa: ANN001
        return _apply(orig_draw(cls, rng, n=n, pd=pd), pd)

    integ.NetworkInitDraws.draw = classmethod(draw)  # type: ignore[method-assign]
    # Seeds with MATLAB-exported draws (tests/fixtures/plant_init_seed*.npz) skip draw().
    orig_cached = pb.load_cached_init_draws
    pb.load_cached_init_draws = lambda seed, **kw: _apply(orig_cached(seed, **kw), 1)  # exported at pd=1  # type: ignore[assignment]

    orig_loop = integ.run_cbgt_loop
    iappgpe = knobs.get("iappgpe")

    def loop(*args):  # noqa: ANN002
        a = list(args)
        if pd_eff is not None:
            a[3] = float(pd_eff)
        if iappgpe is not None:
            a[-2] = float(iappgpe)
        return orig_loop(*a)

    if pd_eff is not None or iappgpe is not None:
        integ.run_cbgt_loop = loop  # type: ignore[assignment]


def mean_psd(spikes: list[np.ndarray], dt_ms: float, seg_s: float) -> tuple[np.ndarray, np.ndarray]:
    params = SpectrumParams.from_dt_ms(dt_ms)
    psds, f = [], None
    for s in spikes:
        p, f = multitaper_psd_point_process(np.asarray(s), params, segment_duration_s=seg_s)
        psds.append(p)
    return f, np.mean(np.vstack(psds), axis=0)


def peak(f: np.ndarray, p: np.ndarray, lo: float = 1.0, hi: float = 50.0, smooth_hz: float = 1.5) -> float:
    dx = float(np.median(np.diff(f)))
    k = int(round(smooth_hz / dx))
    if k > 1:
        pad = np.pad(p, (k // 2, k - 1 - k // 2), mode="edge")
        p = np.convolve(pad, np.ones(k) / k, mode="valid")
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])])


def wmean(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[m]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="knob=value (repeatable)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pd", type=int, default=1)
    ap.add_argument("--dbs-hz", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    knobs = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    seeds = tuple(int(s) for s in args.seeds.split(","))
    cfg = PlantConfig(pd=args.pd)
    for f in ("ggith", "iappth_baseline"):
        if f in knobs:
            cfg = replace(cfg, **{f: knobs.pop(f)})
    install_patches(knobs, cfg.dt_ms)
    dbs = DbsSpec.from_frequency_hz(args.dbs_hz) if args.dbs_hz > 0 else DbsSpec.none()

    pops = {"gpi": [], "stn": [], "gpe": [], "str_indr": [], "str_dr": [], "cor_e": [], "cor_i": [], "th": []}
    rates = {k: [] for k in pops}
    t0 = time.monotonic()
    plant = PythonPlant(cfg)
    for seed in seeds:
        r = plant.reset(seed=seed).integrate(
            args.duration_s, dbs, record_bg_spikes=True, record_cor_spikes=True, record_th_spikes=True
        )
        n = cfg.neurons_per_region
        trains = {
            "gpi": r.gpi_spikes,
            "stn": r.info["stn_spikes"],
            "gpe": r.info["gpe_spikes"],
            "str_indr": r.info["str_indr_spikes"],
            "str_dr": r.info["str_dr_spikes"],
            "cor_e": r.info["cor_spikes"][:n],
            "cor_i": r.info["cor_spikes"][n:],
            "th": r.info["th_spikes"],
        }
        for k, sp in trains.items():
            f, p = mean_psd(sp, cfg.dt_ms, args.duration_s)
            pops[k].append(p)
            rates[k].append(sum(np.asarray(s).size for s in sp) / len(sp) / args.duration_s)
    row = {"label": args.label, "knobs": {**knobs, "ggith": cfg.ggith, "iappth_baseline": cfg.iappth_baseline}, "pd": args.pd, "dbs_hz": args.dbs_hz, "seeds": list(seeds)}
    for k, ps in pops.items():
        p = np.mean(np.vstack(ps), axis=0)
        row[f"{k}_peak_hz"] = peak(f, p)
        row[f"{k}_rate_hz"] = float(np.mean(rates[k]))
        if k == "gpi":
            row["gpi_beta_mean"] = wmean(f, p, 13.0, 35.0)
            row["gpi_low_mean"] = wmean(f, p, 1.0, 12.0)
            row["gpi_peak_8_35"] = peak(f, p, 8.0, 35.0)
    row["elapsed_s"] = round(time.monotonic() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
