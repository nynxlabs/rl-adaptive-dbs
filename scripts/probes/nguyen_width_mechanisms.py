#!/usr/bin/env python3
"""Probe: could pulse shape or the integration step make pulse width matter? (Nguyen Fig 5/6)

On the faithful Kumaravelu plant (every STN neuron gets the full DBS current) wider pulses
barely change alpha-beta past ~0.3 ms, so a trained agent has no reason to widen the pulse
the way the paper's does. This probe holds fixed DBS settings with no agent (as the plant-map
probe ``nguyen_fig5_fig7_plant_map.py`` does) and changes one paper-silent detail at a time:

* pulse shape: charge-balanced biphasic pulses -- a depolarizing (cathodic) phase of the set
  width and amplitude, an optional interphase gap, then an anodic phase of equal charge
  (amplitude A/R for R x the width; R = 1 symmetric, R = 10 close to passive recharge);
* integration step ``dt`` of the plant (reference 0.01 ms), with three ways of putting a pulse
  on a coarse grid: ``round`` (pulse width rounded to whole steps, at least one -- what the
  plant's pulse train does), ``charge`` (whole steps plus one partial-amplitude step carrying
  the remaining charge) and ``hold`` (each pulse starts at its exact time 1/f and a sample is
  on when its sample time falls inside the pulse, i.e. sample-and-hold of the continuous
  waveform; pulses narrower than ``dt`` can fall between samples).

Variants are comma-separated ``key=value`` lists: ``rho`` (STN recruitment spread, 1 =
faithful), ``shape`` (``mono`` | ``bi``), ``gap`` (ms), ``ratio`` (R), ``dt`` (ms),
``sampling`` (``round`` | ``charge`` | ``hold``). ``mono`` alone is the baseline.
Alpha-beta always uses the reference spectrum grid (0.01 ms; spike times are in seconds), so
a coarse ``dt`` changes only the dynamics; ``ab_native`` in the output is the value on the
coarse grid. See docs/figures/nguyen/5.md § Pulse shape and integration step. Run from the
repo root:

    RL_DBS_MAX_THREADS=1 .venv/bin/python scripts/probes/nguyen_width_mechanisms.py \\
        --out /tmp/width_mech.json --variants mono rho=3 shape=bi,gap=0 dt=0.1
"""

from __future__ import annotations

import os

for _var in ("RL_DBS_MAX_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nguyen_fig5_fig7_plant_map import _hold_action

import controllers.snn.adapter as adapter_mod
from controllers.snn.adapter import NguyenEnvAdapter, observed_populations
from controllers.snn.config import fig4_nguyen_config
from controllers.snn.reward import alpha_beta_power
from envs.plant.config import PlantConfig
from envs.plant.dbs import ContinuousPulseTrain
from envs.plant import PythonPlant

REFERENCE_DT_MS = 0.01
PAPER_INIT = (40.0, 300.0, 0.3)
_NATIVE_AB: list[float] = []


def _reference_grid_alpha_beta(gpi_spikes, *, duration_s: float, dt_ms: float = REFERENCE_DT_MS) -> float:
    """Alpha-beta on the reference 0.01 ms spectrum grid; the coarse-grid value is kept aside."""
    if abs(dt_ms - REFERENCE_DT_MS) > 1e-12:
        _NATIVE_AB.append(alpha_beta_power(gpi_spikes, duration_s=duration_s, dt_ms=dt_ms))
    return alpha_beta_power(gpi_spikes, duration_s=duration_s, dt_ms=REFERENCE_DT_MS)


adapter_mod.alpha_beta_power = _reference_grid_alpha_beta


class ShapedPulseTrain:
    """Phase-continuous STN pulse train with a pulse template (``ContinuousPulseTrain`` clock).

    Each pulse is a template of samples starting at its onset sample; pulses that run past a
    segment end finish in the next segment. Overlapping templates add (only long anodic phases
    at high frequency overlap). ``shape='mono'`` with ``sampling='round'`` reproduces
    ``ContinuousPulseTrain`` exactly (``--selftest``).
    """

    def __init__(
        self,
        dt_ms: float = REFERENCE_DT_MS,
        *,
        shape: str = "mono",
        gap_ms: float = 0.0,
        ratio: float = 1.0,
        sampling: str = "round",
    ) -> None:
        if shape not in ("mono", "bi") or sampling not in ("round", "charge", "hold"):
            raise ValueError(f"bad shape/sampling {shape!r}/{sampling!r}")
        self.dt_ms, self.shape, self.gap_ms, self.ratio, self.sampling = (
            float(dt_ms), shape, float(gap_ms), float(ratio), sampling,
        )
        self.reset()

    def reset(self) -> None:
        self._origin = 0
        self._pulses: list[tuple[int, np.ndarray]] = []
        self._last_onset_ms: float | None = None

    def _phases_ms(self, pw: float, amp: float) -> list[tuple[float, float, float]]:
        """(start ms, duration ms, amplitude) of each phase relative to the pulse onset."""
        phases = [(0.0, pw, amp)]
        if self.shape == "bi":
            phases.append((pw + self.gap_ms, self.ratio * pw, -amp / self.ratio))
        return phases

    def _template(self, pw: float, amp: float, onset_ms: float) -> tuple[int, np.ndarray]:
        """Onset sample and template samples for one pulse."""
        dt = self.dt_ms
        if self.sampling == "hold":
            # Sample k (time k*dt) takes the continuous waveform's value at that time.
            first = math.ceil(onset_ms / dt - 1e-9)
            end_ms = onset_ms + max(s + d for s, d, _ in self._phases_ms(pw, amp))
            n = max(0, math.ceil(end_ms / dt - 1e-9) - first)
            t = (first + np.arange(n)) * dt - onset_ms
            tpl = np.zeros(n)
            for s, d, a in self._phases_ms(pw, amp):
                tpl[(t >= s - 1e-9) & (t < s + d - 1e-9)] += a
            return first, tpl
        onset = round(onset_ms / dt)
        chunks = []
        for s, d, a in self._phases_ms(pw, amp):
            lead = round(s / dt) - sum(len(c) for c in chunks)
            chunks.append(np.zeros(max(0, lead)))
            if self.sampling == "round":
                chunks.append(np.full(max(1, round(d / dt)), a))
            else:  # charge: whole steps, then one step carrying the leftover charge
                whole = math.floor(d / dt + 1e-9)
                frac = d / dt - whole
                chunks.append(np.full(whole, a))
                if frac > 1e-9:
                    chunks.append(np.array([a * frac]))
        return onset, np.concatenate(chunks)

    def segment(self, *, frequency_hz: float, pulse_width_ms: float, amplitude: float, tmax_ms: float):
        dt = self.dt_ms
        n = round(tmax_ms / dt)
        g0 = self._origin
        idbs = np.zeros(n + 1)
        pulses = list(self._pulses)
        last = self._last_onset_ms
        if frequency_hz > 0.0 and amplitude > 0.0:
            if self.sampling == "hold":
                isi_ms = 1000.0 / frequency_hz
                onset_ms = g0 * dt if last is None else max(last + isi_ms, (g0 + 1) * dt - 1e-9)
            else:
                isi = round((1000.0 / frequency_hz) / dt)
                isi_ms = isi * dt
                onset_ms = g0 * dt if last is None else max(
                    round(last / dt) + isi, g0 + 1) * dt
            while onset_ms <= (g0 + n) * dt + 1e-9:
                pulses.append(self._template(pulse_width_ms, amplitude, onset_ms))
                last = onset_ms
                onset_ms += isi_ms
        else:
            last = None
        for onset, tpl in pulses:
            lo, hi = max(onset, g0), min(onset + len(tpl), g0 + n + 1)
            if hi > lo:
                idbs[lo - g0 : hi - g0] += tpl[lo - onset : hi - onset]
        keep = [(o, t) for o, t in pulses if o + len(t) > g0 + n]
        return idbs, {"origin": g0 + n, "pulses": keep, "last": last}

    def commit(self, state) -> None:
        self._origin, self._pulses, self._last_onset_ms = state["origin"], state["pulses"], state["last"]


def parse_variant(text: str) -> dict:
    v = {"rho": 1.0, "shape": "mono", "gap": 0.0, "ratio": 1.0, "dt": REFERENCE_DT_MS, "sampling": "round"}
    if text != "mono":
        for item in text.split(","):
            key, val = item.split("=")
            v[key] = val if key in ("shape", "sampling") else float(val)
    return v


def run(variant: str, freq: float, amp: float, pw: float, seed: int, steps: int) -> dict:
    v = parse_variant(variant)
    cfg = replace(fig4_nguyen_config(seed=seed), max_episode_steps=steps, stn_dbs_spread=v["rho"])
    env = NguyenEnvAdapter(config=cfg)
    env._plant.close()
    env._plant = PythonPlant(PlantConfig(dt_ms=v["dt"], stn_dbs_spread=v["rho"]))
    plain = v["shape"] == "mono" and v["sampling"] == "round"
    env._pulses = (
        ContinuousPulseTrain(dt_ms=v["dt"]) if plain
        else ShapedPulseTrain(v["dt"], shape=v["shape"], gap_ms=v["gap"], ratio=v["ratio"], sampling=v["sampling"])
    )
    pops = observed_populations(cfg)
    hold = _hold_action(cfg)
    t0 = time.time()
    rows = []
    try:
        _NATIVE_AB.clear()
        _obs, info = env.reset(seed=seed)
        rows.append({"alpha_beta": info["alpha_beta"], "pops": info["cbgt_spike_events_by_population"]})
        env._dbs.frequency_hz, env._dbs.amplitude, env._dbs.pulse_width_ms = freq, amp, pw
        for _ in range(steps):
            _obs, _r, _te, _tr, info = env.step(hold)
            rows.append({"alpha_beta": info["alpha_beta"], "pops": info["cbgt_spike_events_by_population"],
                         "guard": bool(info["plant_guard"])})
    finally:
        env.close()
    ab = np.array([r["alpha_beta"] for r in rows])
    stn = np.array([r["pops"][pops.index("stn")] for r in rows])
    gpi = np.array([r["pops"][pops.index("gpi")] for r in rows])
    return {
        "variant": variant, "freq": freq, "amp": amp, "pw": pw, "seed": seed,
        "ab_6_25": float(ab[6:].mean()), "stn_6_25": float(stn[6:].mean()), "gpi_6_25": float(gpi[6:].mean()),
        "guards": int(sum(r.get("guard", False) for r in rows)),
        "ab_native_6_25": float(np.mean(_NATIVE_AB[6:])) if _NATIVE_AB else None,
        "rows": rows, "seconds": time.time() - t0,
    }


def _job(args):
    return run(*args)


def selftest() -> None:
    """Mono/round must match ContinuousPulseTrain sample for sample; print biphasic charge."""
    for dt in (0.01, 0.1):
        a, b = ContinuousPulseTrain(dt_ms=dt), ShapedPulseTrain(dt)
        for f, pw, amp in ((40, 0.3, 300), (130, 1.1, 262), (60, 0.6, 262), (80, 0.1, 262)):
            for _ in range(3):
                xa, sa = a.segment(frequency_hz=f, pulse_width_ms=pw, amplitude=amp, tmax_ms=100.0)
                xb, sb = b.segment(frequency_hz=f, pulse_width_ms=pw, amplitude=amp, tmax_ms=100.0)
                assert np.array_equal(xa, xb), (dt, f, pw)
                a.commit(sa)
                b.commit(sb)
    for kw in ({"shape": "bi"}, {"shape": "bi", "gap_ms": 0.1}, {"shape": "bi", "ratio": 10.0},
               {"sampling": "hold"}, {"sampling": "charge"}):
        dt = 0.25 if "sampling" in kw else 0.01
        tr = ShapedPulseTrain(dt, **kw)
        tot = []
        for _ in range(5):
            x, s = tr.segment(frequency_hz=80, pulse_width_ms=0.3, amplitude=262, tmax_ms=100.0)
            tr.commit(s)
            tot.append(x[1:])  # sample 0 repeats the previous segment's last sample
        x = np.concatenate(tot)
        print(kw, f"dt={dt}: net charge {x.sum() * dt:.3f}, cathodic {x[x > 0].sum() * dt:.1f} nC/cm2 "
              f"over 0.5 s (40 pulses x 262 x 0.3 = {40 * 262 * 0.3:.0f})")
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variants", nargs="+", default=["mono"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--freqs", type=float, nargs="+", default=[60, 80, 130])
    ap.add_argument("--pws", type=float, nargs="+", default=[0.1, 0.3, 0.6, 0.8, 1.1])
    ap.add_argument("--amp", type=float, default=262.0)
    ap.add_argument("--no-init", action="store_true", help="skip the paper's initial setting")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    settings = [] if args.no_init else [PAPER_INIT]
    settings += [(f, args.amp, pw) for f in args.freqs for pw in args.pws]
    jobs = [(v, f, a, pw, s, args.steps) for v in args.variants for f, a, pw in settings for s in args.seeds]
    out = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(_job, jobs):
            print(f"{r['variant']:28s} f={r['freq']:5.1f} amp={r['amp']:4.0f} pw={r['pw']:.2f} seed={r['seed']} "
                  f"ab6-25={r['ab_6_25']:6.1f} stn={r['stn_6_25']:5.1f} gpi={r['gpi_6_25']:5.1f} "
                  f"guards={r['guards']} ({r['seconds']:.0f}s)", flush=True)
            out.append(r)
            if args.out:
                args.out.write_text(json.dumps(out))
    print("\nalpha-beta, steps 6-25, mean over seeds (STN spikes per step)")
    for v in args.variants:
        print(f"[{v}]")
        for f, a, pw in settings:
            rs = [r for r in out if r["variant"] == v and (r["freq"], r["amp"], r["pw"]) == (f, a, pw)]
            per_seed = " / ".join(f"{r['ab_6_25']:.0f}" for r in rs)
            print(f"  {f:5.1f} Hz {a:4.0f} x {pw:.2f}: {np.mean([r['ab_6_25'] for r in rs]):6.1f} "
                  f"(seeds {per_seed}) stn {np.mean([r['stn_6_25'] for r in rs]):5.1f}")


if __name__ == "__main__":
    main()
