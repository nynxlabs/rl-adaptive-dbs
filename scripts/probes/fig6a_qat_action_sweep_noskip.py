#!/usr/bin/env python3
"""Sweep open-loop actions with skip_regular=False (41-pattern alphabet)."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_PLOT = _ROOT / "scripts" / "figures" / "papers" / "mehregan" / "6a" / "plot.py"
_FIG2A = _ROOT / "scripts" / "figures" / "papers" / "mehregan" / "2a" / "plot.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=str, default="0-10,20")
    parser.add_argument("--target", type=float, default=499.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    p6a = _load(_PLOT, "p6a")
    p2a = _load(_FIG2A, "p2a")
    from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
    from envs.plant import PlantConfig, PythonPlant

    actions: list[int] = []
    for part in args.actions.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            actions.extend(range(int(lo), int(hi) + 1))
        else:
            actions.append(int(part))

    times = p2a.sample_times(p2a.STEP_S, duration_s=p2a.DISPLAY_S)
    t = np.asarray(times)
    post_mask = t >= 2.0
    baseline = 503.19
    band_lo = 0.85 * baseline
    band_hi = 1.05 * baseline
    print(f"skip_regular=False target={args.target:.1f} band=[{band_lo:.0f},{band_hi:.0f}]", flush=True)

    best: tuple[float, int] | None = None
    for action in actions:
        plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=p6a.PAPER_DT_MS))
        alphabet = BurstPatternAlphabet(
            mean_hz=p6a.MEAN_HZ,
            step_duration_s=p6a.TRAILING_RL_STEP_S,
            dt_ms=float(p6a.PAPER_DT_MS),
            skip_regular=False,
        )
        if action < 0 or action >= alphabet.n_actions:
            print(f"action {action:2d}: skip (out of range n={alphabet.n_actions})", flush=True)
            continue
        seg = [action] * p6a.TRAILING_STIM_STEPS
        trace = p6a._trailing_condition_trace(
            plant,
            seed=args.seed,
            label=f"a{action}",
            segment_actions=seg,
            alphabet=alphabet,
            times=times,
        )
        plant.close()
        post = float(np.mean(np.asarray(trace)[post_mask]))
        band_ok = band_lo <= post <= band_hi
        dist = abs(post - args.target)
        print(f"action {action:2d}: post={post:6.1f} band_ok={band_ok}", flush=True)
        if best is None or dist < best[0]:
            best = (dist, action)

    if best:
        print(f"closest: action {best[1]} (|post-target|={best[0]:.1f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
