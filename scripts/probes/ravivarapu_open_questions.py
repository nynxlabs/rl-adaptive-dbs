#!/usr/bin/env python3
"""Ravivarapu plant probes for the open author questions (no RL training).

Experiments (each prints a table and lands in one JSON):

``dt``         Q18 — paper's 0.02 ms integration step vs our 0.01 ms: untreated raw
               P_beta and full-window 50 / 30 Hz stim/untreated ratios.
``stim30``     Q11 / Q18 — can any DBS amplitude / pulse width make 30 Hz lower GPi
               beta (while 50 Hz still suppresses)? Full-window carrier, 150 ms shots.
``continuous`` Q1 / Q10 — on one continuous plant simulation per episode, is
               always-on still the best fixed action pattern under Eq. (7)? If an
               intermittent pattern beat it, the continuous reading could make the
               trained variants differ.
``table2``     Q14 — Table II protocol readings for the always-on policy (both trained
               actors stimulate every step): window re-prefilled at each reseed vs
               carried over, n_obs, plant mode. Each reading is fitted to
               avg(n) = F + A / n, the form the paper's SEA-DBS row follows.
``psd_scale``  Q3 — absolute scale of our point-process multitaper PSD: a Poisson
               train of rate lambda has a flat two-sided density lambda, so
               a one-sided PSD doubles every band power.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_open_questions.py \
      --experiments dt stim30 continuous psd_scale \
      --out artifacts/figures/papers/ravivarapu/probes/open_questions.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.reward import sea_dbs_reward
from envs.plant.biomarkers import SpectrumParams, band_power, multitaper_psd_point_process
from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant import PythonPlant


class _CustomStimAdapter(SEA_DBSEnvAdapter):
    """Adapter whose a = 1 is a full-window train with a chosen amplitude / pulse width."""

    def __init__(self, *, amplitude: float, pulse_width_ms: float, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._amp = float(amplitude)
        self._pw = float(pulse_width_ms)

    def _dbs_spec_for_action(self, action: int, *, duration_s: float | None = None) -> DbsSpec:
        if int(action) == 0:
            return DbsSpec.none()
        ms = (self.config.integration_duration_s if duration_s is None else duration_s) * 1000.0
        idbs = create_dbs_current(
            self._carrier_hz,
            tmax_ms=ms,
            dt_ms=self.config.plant_dt_ms,
            pulse_width_ms=self._pw,
            amplitude=self._amp,
        )
        return DbsSpec(pick_dbs_freq=2, idbs=idbs, mean_hz=self._carrier_hz)


def _ratios(env: SEA_DBSEnvAdapter, seeds: list[int]) -> dict[str, list[float]]:
    untreated, stim = [], []
    for seed in seeds:
        _o, info = env.reset(seed=seed)
        untreated.append(float(info["p_beta_raw"]))
        _o, _r, _t, _tr, step = env.step(1)
        stim.append(float(step["p_beta_raw"]))
    u, s = np.asarray(untreated), np.asarray(stim)
    return {"untreated": untreated, "stim": stim, "ratio": (s / u).tolist()}


def _fmt(name: str, rec: dict[str, list[float]]) -> str:
    u, r = np.asarray(rec["untreated"]), np.asarray(rec["ratio"])
    return f"{name:28s} untreated {u.mean():6.1f}±{u.std():5.1f}  ratio {r.mean():.3f}±{r.std():.3f}"


def exp_dt(seeds: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for dt in (0.01, 0.02):
        for hz in (50.0, 30.0):
            cfg = SEADBSConfig(plant_dt_ms=dt, carrier_hz=hz)
            env = SEA_DBSEnvAdapter(config=cfg, plant=PythonPlant(PlantConfig(dt_ms=dt)))
            env.set_carrier_hz(hz)
            try:
                rec = _ratios(env, seeds)
            finally:
                env.close()
            name = f"dt{dt}_{int(hz)}hz"
            out[name] = rec
            print(_fmt(name, rec), flush=True)
    return out


def exp_stim30(seeds: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for amp in (50.0, 100.0, 200.0, 300.0, 600.0, 1000.0):
        for pw in (0.06, 0.1, 0.3):
            for hz in (30.0, 50.0):
                cfg = SEADBSConfig(carrier_hz=hz)
                env = _CustomStimAdapter(config=cfg, amplitude=amp, pulse_width_ms=pw)
                env.set_carrier_hz(hz)
                try:
                    rec = _ratios(env, seeds)
                finally:
                    env.close()
                name = f"amp{int(amp)}_pw{pw}_{int(hz)}hz"
                out[name] = rec
                print(_fmt(name, rec), flush=True)
    return out


PATTERNS: dict[str, list[int]] = {
    "on": [1],
    "off": [0],
    "1on1off": [1, 0],
    "2on1off": [1, 1, 0],
    "3on1off": [1, 1, 1, 0],
    "5on5off": [1] * 5 + [0] * 5,
    "5on1off": [1] * 5 + [0],
}


def exp_continuous(seeds: list[int], steps: int = 30) -> dict[str, object]:
    out: dict[str, object] = {}
    for mode in ("continuous", "disconnected"):
        cfg = replace(SEADBSConfig(), plant_integration_mode=mode)
        for name, unit in PATTERNS.items():
            acts = (unit * steps)[:steps]
            rewards, means = [], []
            for seed in seeds:
                env = SEA_DBSEnvAdapter(config=cfg)
                env.set_carrier_hz(cfg.carrier_hz)
                try:
                    _o, info = env.reset(seed=seed)
                    trace, rsum = [float(info["p_beta_norm"])], 0.0
                    for a in acts:
                        _o, r, _t, _tr, step = env.step(a)
                        trace.append(float(step["p_beta_norm"]))
                        rsum += float(r)
                finally:
                    env.close()
                rewards.append(rsum / steps)
                means.append(trace)
            m = np.asarray(means)
            out[f"{mode}_{name}"] = {
                "mean_reward_per_step": rewards,
                "pbar_mean_trace": m.mean(axis=0).tolist(),
                "duty": float(np.mean(acts)),
            }
            print(
                f"{mode:12s} {name:8s} duty {np.mean(acts):.2f}  reward/step "
                f"{np.mean(rewards):+.3f}±{np.std(rewards):.3f}  P̄ ep-mean {m[:, 1:].mean():.3f}"
                f"  P̄ step10 {m[:, min(10, steps)].mean():.3f}  step30 {m[:, -1].mean():.3f}",
                flush=True,
            )
    return out


PAPER_TABLE2 = {"baseline": {10: 364, 20: 366, 50: 332, 75: 328}, "sea": {10: 326, 20: 309, 50: 304, 75: 302}}


def _fit_f_a(avg: dict[int, float]) -> tuple[float, float]:
    ns = np.array(sorted(avg), dtype=float)
    y = np.array([avg[int(n)] for n in ns])
    design = np.stack([np.ones_like(ns), 1.0 / ns], axis=1)
    (f, a), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(f), float(a)


def exp_table2(horizon: int = 300, seed0: int = 0) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, f_paper in PAPER_TABLE2.items():
        f, a = _fit_f_a({k: float(v) for k, v in f_paper.items()})
        out[f"paper_{name}_fit"] = {"F": f, "A": a}
        print(f"paper {name:8s} fit F {f:6.1f}  A {a:6.0f}", flush=True)
    readings = [
        ("reprefill_n10", "disconnected", 10, True),
        ("reprefill_n5", "disconnected", 5, True),
        ("reprefill_n4", "disconnected", 4, True),
        ("reprefill_n3", "disconnected", 3, True),
        ("carryover_n10", "disconnected", 10, False),
        ("reprefill_n10_cont", "continuous", 10, True),
        ("carryover_n10_cont", "continuous", 10, False),
    ]
    for name, mode, n_obs, reprefill in readings:
        cfg = replace(SEADBSConfig(), plant_integration_mode=mode, n_obs=n_obs)
        avg: dict[int, float] = {}
        for n in (10, 20, 50, 75):
            env = SEA_DBSEnvAdapter(config=replace(cfg, max_episode_steps=horizon + 1))
            env.set_carrier_hz(cfg.carrier_hz)
            vals: list[float] = []
            try:
                for block in range(horizon // n):
                    if block == 0 or reprefill:
                        env.reset(seed=seed0 + block)
                    else:
                        # New environment seed, same observation history (Eq. (4) window).
                        env._plant.reset(seed=seed0 + block)  # noqa: SLF001
                    for _ in range(n):
                        _o, _r, _t, _tr, step = env.step(1)
                        vals.append(float(step["p_beta_norm"]) * 1000.0)
            finally:
                env.close()
            avg[n] = float(np.mean(vals))
        f, a = _fit_f_a(avg)
        out[name] = {"avg_psd": avg, "F": f, "A": a}
        print(
            f"{name:20s} " + " ".join(f"n{n}:{avg[n]:6.1f}" for n in avg) + f"   fit F {f:6.1f}  A {a:6.0f}",
            flush=True,
        )
    return out


def exp_psd_scale(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    out: dict[str, object] = {}
    for rate in (10.0, 40.0):
        for dur in (0.15, 1.0):
            pw = []
            for _ in range(200):
                n = rng.poisson(rate * dur)
                spikes = np.sort(rng.uniform(0.0, dur, n))
                psd, freqs = multitaper_psd_point_process(spikes, SpectrumParams.from_dt_ms(0.01), segment_duration_s=dur)
                pw.append(band_power(psd, freqs, 13.0, 35.0))
            mean_density = float(np.mean(pw)) / 22.0
            out[f"rate{int(rate)}_dur{dur}"] = {"band_power": float(np.mean(pw)), "density": mean_density}
            print(
                f"Poisson {rate:4.0f} Hz, {dur:.2f} s: 13–35 Hz power {np.mean(pw):7.1f} "
                f"→ density {mean_density:6.2f} (two-sided would be {rate:.0f})",
                flush=True,
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["dt", "stim30", "continuous", "table2", "psd_scale"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--continuous-seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: dict[str, object] = {"seeds": args.seeds}
    if "dt" in args.experiments:
        out["dt"] = exp_dt(args.seeds)
    if "stim30" in args.experiments:
        out["stim30"] = exp_stim30(args.seeds)
    if "continuous" in args.experiments:
        out["continuous"] = exp_continuous(args.continuous_seeds)
    if "table2" in args.experiments:
        out["table2"] = exp_table2()
    if "psd_scale" in args.experiments:
        out["psd_scale"] = exp_psd_scale()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
