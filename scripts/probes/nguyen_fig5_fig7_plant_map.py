#!/usr/bin/env python3
"""Open-loop plant map for Nguyen Figs 5 and 7.

Holds one DBS setting for a 25-step episode under the Fig 4 recipe's plant
(carried pulse train, 8 observed populations) and records alpha-beta power and
spike events per population per 100 ms step. Answers two questions:

* Fig 5: which spike count stays flat as the policy raises frequency?
* Fig 7: does alpha-beta rise again at the high frequency / charge the policy
  reaches late in an evaluation episode, and what is alpha-beta at step 0?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from controllers.snn.actions import decode_joint_action  # noqa: E402
from controllers.snn.adapter import NguyenEnvAdapter  # noqa: E402
from controllers.snn.config import fig4_nguyen_config  # noqa: E402

POPS = ("cor_exc", "cor_inh", "str_dr", "str_indr", "stn", "gpe", "gpi", "th")


def _hold_action(cfg) -> object:
    if cfg.action_scheme == "joint":
        return next(i for i in range(cfg.n_action_outputs) if not decode_joint_action(i).any())
    return np.array([1, 1, 1], dtype=np.int64)


def _pop_counts(env: NguyenEnvAdapter, info: dict) -> list[int]:
    # Re-derive per-population counts from the trains the adapter observed.
    trains = env._last_trains  # set by the patched encoder hook below
    n = env.config.neurons_per_region
    return [int(sum(np.asarray(t).size for t in trains[i * n : (i + 1) * n])) for i in range(len(POPS))]


def run(freq: float, amp: float, pw: float, seed: int, steps: int) -> dict:
    cfg = replace(fig4_nguyen_config(seed=seed), max_episode_steps=steps)
    env = NguyenEnvAdapter(config=cfg)
    orig = env._observed_spike_trains

    def hook(result):
        trains = orig(result)
        env._last_trains = trains
        return trains

    env._observed_spike_trains = hook
    hold = _hold_action(cfg)
    try:
        _obs, info = env.reset(seed=seed)
        rows = [{"alpha_beta": info["alpha_beta"], "pops": _pop_counts(env, info), "dbs": "init"}]
        env._dbs.frequency_hz, env._dbs.amplitude, env._dbs.pulse_width_ms = freq, amp, pw
        for _ in range(steps):
            _obs, _r, _te, _tr, info = env.step(hold)
            rows.append({"alpha_beta": info["alpha_beta"], "pops": _pop_counts(env, info)})
    finally:
        env.close()
    return {"freq": freq, "amp": amp, "pw": pw, "seed": seed, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--freqs", type=float, nargs="+", default=[40, 50, 60, 70, 75, 80, 90])
    ap.add_argument("--charges", default="300:0.3,380:0.7,450:1.1")
    args = ap.parse_args()
    charges = [tuple(float(v) for v in c.split(":")) for c in args.charges.split(",")]
    out = []
    for f in args.freqs:
        for amp, pw in charges:
            for s in args.seeds:
                r = run(f, amp, pw, s, args.steps)
                ab = [row["alpha_beta"] for row in r["rows"]]
                print(f"f={f:5.1f} amp={amp:4.0f} pw={pw:.2f} seed={s} ab0={ab[0]:6.1f} "
                      f"ab1-5={np.mean(ab[1:6]):6.1f} ab6-25={np.mean(ab[6:]):6.1f}", flush=True)
                out.append(r)
                args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
