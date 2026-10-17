#!/usr/bin/env python3
"""Ravivarapu 2 ms steps (``step_mode="trailing"``): open-loop traces, no training.

For each trailing P_beta window and pulse reading, rolls fixed action patterns
(always-on, always-off, alternate) over a 30-step episode on plant seeds and
reports: the untreated raw P_beta at reset (calibrates ``observation_scale`` so
untreated sits at the paper's 0.4617, the rule behind the 150 ms shot's 472),
the Eq. (5) mean P̄ at steps 0 / 10 / 30, the episode mean of P̄ (what Fig 4a
plots), and the mean Eq. (7) reward per step. ``carrier`` runs at 50 and 30 Hz.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_trailing_steps.py \
      --windows 0.1 0.15 0.3 --seeds 0 1 2 3 4 5 6 7 8 9 \
      --out artifacts/figures/papers/ravivarapu/probes/trailing_steps.json
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

PAPER_UNTREATED = 0.4617  # 217.9 / 472: the level the 150 ms convention maps untreated to
PATTERNS = {"on": [1], "off": [0], "alt": [1, 0]}


def _rollout(cfg: SEADBSConfig, seed: int, acts: list[int], hz: float) -> tuple[float, list[float]]:
    env = SEA_DBSEnvAdapter(config=cfg)
    env.set_carrier_hz(hz)
    try:
        _o, info = env.reset(seed=seed)
        raw0 = float(info["p_beta_raw"])
        pbar = [float(info["p_beta_norm"])]
        for a in acts:
            _o, _r, _t, _tr, step = env.step(a)
            pbar.append(float(step["p_beta_norm"]))
    finally:
        env.close()
    return raw0, pbar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=float, nargs="+", default=[0.1, 0.15, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: dict[str, object] = {"seeds": args.seeds, "steps": args.steps}
    for w in args.windows:
        base = replace(SEADBSConfig(), step_mode="trailing", trailing_window_s=w, max_episode_steps=args.steps + 1)
        # Calibrate the scale on untreated reset levels (scale-free raw values).
        raws = [_rollout(replace(base, observation_scale=1.0), s, [], 50.0)[0] for s in args.seeds]
        scale = float(np.mean(raws)) / PAPER_UNTREATED
        cfg_w = replace(base, observation_scale=scale)
        rec: dict[str, object] = {"untreated_raw": raws, "observation_scale": scale}
        print(f"W {w * 1000:.0f} ms: untreated raw {np.mean(raws):.1f} ± {np.std(raws):.1f} → scale {scale:.1f}", flush=True)
        for mode, hz in (("carrier", 50.0), ("carrier", 30.0), ("per_step", 50.0)):
            cfg = replace(cfg_w, step_pulse_mode=mode)
            for name, unit in PATTERNS.items():
                if name != "on" and not (mode == "carrier" and hz == 50.0):
                    continue
                acts = (unit * args.steps)[: args.steps]
                traces = np.array([_rollout(cfg, s, acts, hz)[1] for s in args.seeds])
                mean = traces.mean(axis=0)
                rewards = [
                    float(np.mean([sea_dbs_reward(v) for v in tr[1:]])) for tr in traces
                ]
                key = f"{mode}_{int(hz)}hz_{name}"
                rec[key] = {"pbar_mean_trace": mean.tolist(), "reward_per_step": rewards}
                print(
                    f"  {key:20s} P̄ step0 {mean[0]:.3f} step10 {mean[min(10, args.steps)]:.3f} "
                    f"step30 {mean[-1]:.3f}  ep-mean {traces[:, 1:].mean():.3f}  "
                    f"10-step drop {mean[0] - mean[min(10, args.steps)]:+.3f}  "
                    f"reward/step {np.mean(rewards):+.3f}",
                    flush=True,
                )
        out[f"w{int(round(w * 1000))}"] = rec
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
