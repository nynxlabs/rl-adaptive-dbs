#!/usr/bin/env python3
"""Sweep burst open-loop actions for Fig 6a QAT late-sustain gate."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-action", type=int, default=1)
    parser.add_argument("--max-action", type=int, default=40)
    args = parser.parse_args()

    p6a = _load(_ROOT / "scripts/figures/papers/mehregan/6a/plot.py", "p6a")
    p2a = _load(_ROOT / "scripts/figures/papers/mehregan/2a/plot.py", "p2a")
    from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
    from envs.plant import PlantConfig, PythonPlant

    times = p2a.sample_times(p2a.STEP_S, duration_s=p2a.DISPLAY_S)
    t = np.asarray(times)
    hits: list[int] = []
    for action in range(args.min_action, args.max_action + 1):
        plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=p6a.PAPER_DT_MS))
        alphabet = BurstPatternAlphabet(
            mean_hz=45.0,
            step_duration_s=p6a.TRAILING_RL_STEP_S,
            dt_ms=float(p6a.PAPER_DT_MS),
            skip_regular=True,
        )
        if action < 0 or action >= alphabet.n_actions:
            plant.close()
            print(f"a{action:02d}: skip", flush=True)
            continue
        trace = np.asarray(
            p6a._trailing_condition_trace(
                plant,
                seed=args.seed,
                label=f"a{action}",
                segment_actions=[action] * p6a.TRAILING_STIM_STEPS,
                alphabet=alphabet,
                times=times,
            ),
            dtype=float,
        )
        plant.close()
        early = float(trace[(t >= 2) & (t <= 8)].mean())
        late = float(trace[(t >= 10) & (t <= 12)].mean())
        end = float(trace[np.argmin(np.abs(t - 12.0))])
        peak = float(trace[t >= 2].max())
        ok = (
            early >= 420.0
            and late >= 0.9 * early
            and (peak - end) <= 60.0
        )
        print(
            f"a{action:02d} early={early:6.1f} late={late:6.1f} end={end:6.1f} "
            f"peak-end={peak - end:5.1f} ok={ok}",
            flush=True,
        )
        if ok:
            hits.append(action)
    print(f"HITS {hits}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
