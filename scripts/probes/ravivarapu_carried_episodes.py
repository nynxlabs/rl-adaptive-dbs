#!/usr/bin/env python3
"""Ravivarapu 2 ms steps with the plant carried across episodes: open-loop Fig 4a.

``step_mode="trailing"`` + ``carry_plant_across_episodes=True`` makes the whole
150-episode run one simulation. This rolls fixed action rules through it and
reports the Fig 4a quantity (episode mean of P̄) over the paper's episode
windows, so we can see whether stimulation accumulated across 60 ms episodes
lowers beta at all, and how fast. Rules: always on, always off, alternate, and a
duty ramp 0.5 → 0.95 over the run (what the ε-greedy Baseline plays).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_carried_episodes.py \
      --windows 0.15 --seeds 0 1 2 \
      --out artifacts/figures/papers/ravivarapu/probes/carried_episodes.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig

SCALES = {0.1: 391.0, 0.15: 459.5, 0.3: 785.6}  # untreated → 0.4617 (trailing_steps probe)
WINDOWS = ((0, 15), (40, 80), (70, 110), (120, 150))


def _rule(name: str, episode: int, step: int, rng: np.random.Generator, n_ep: int) -> int:
    if name == "on":
        return 1
    if name == "off":
        return 0
    if name == "alt":
        return step % 2
    duty = 0.5 + 0.45 * episode / max(1, n_ep - 1)
    return int(rng.random() < duty)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=float, nargs="+", default=[0.15])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--rules", nargs="+", default=["on", "off", "alt", "ramp"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: dict[str, object] = {"seeds": args.seeds, "episodes": args.episodes}
    for w in args.windows:
        cfg = replace(
            SEADBSConfig(),
            step_mode="trailing",
            trailing_window_s=w,
            observation_scale=SCALES[w],
            carry_plant_across_episodes=True,
        )
        for rule in args.rules:
            curves = []
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                env = SEA_DBSEnvAdapter(config=cfg)
                env.set_carrier_hz(cfg.carrier_hz)
                ep_psd = []
                try:
                    for ep in range(args.episodes):
                        _o, info = env.reset(seed=seed + ep)
                        vals = [float(info["p_beta_norm"])]
                        for step in range(cfg.max_episode_steps):
                            _o, _r, _t, _tr, s = env.step(_rule(rule, ep, step, rng, args.episodes))
                            vals.append(float(s["p_beta_norm"]))
                        ep_psd.append(float(np.mean(vals)))
                finally:
                    env.close()
                curves.append(ep_psd)
            c = np.asarray(curves)
            win = [float(c[:, a:b].mean()) for a, b in WINDOWS]
            out[f"w{int(round(w * 1000))}_{rule}"] = {"episode_psd": c.tolist(), "window_means": win}
            print(
                f"W {w * 1000:.0f} ms {rule:5s} early/mid/midlate/late "
                + " / ".join(f"{v:.3f}" for v in win)
                + f"   seed spread late {c[:, 120:150].mean(axis=1).round(3).tolist()}",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
