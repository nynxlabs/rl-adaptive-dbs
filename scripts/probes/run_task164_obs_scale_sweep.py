#!/usr/bin/env python3
"""TASK-164: observation_scale sweep @ 30 Hz L=16 (secondary hypothesis)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

PAPER_DT_MS = 0.02
ARTIFACTS = Path("artifacts/ddpg")
DEFAULT_SCALES = (400.0, 500.0, 1000.0, 2000.0)
MEAN_HZ = 30.0
STATE_LENGTH = 16
SEED = 0
FINAL_OUT = ARTIFACTS / "task160_obs_scale_sweep.json"


def _scale_path(scale: float) -> Path:
    tag = str(int(scale)) if float(scale).is_integer() else str(scale).replace(".", "p")
    return ARTIFACTS / f"task160_obs_scale_sweep_{tag}.json"


def run_scale(scale: float) -> dict:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        observation_scale=scale,
    )
    env = MehreganEnv(plant=PythonPlant(config=plant_cfg), config=env_cfg)
    alphabet = env.alphabet
    if isinstance(alphabet, FixedMeanPatternAlphabet):
        for i in range(alphabet.n_actions):
            alphabet.idbs_for_pattern(i)

    rows: list[tuple[int, float, float]] = []
    try:
        for action in range(env.alphabet.n_actions):
            env.reset(seed=SEED)
            _obs, reward, _t, _tr, info = env.step(action)
            rows.append((action, float(reward), float(info["p_beta_norm"])))
            print(
                f"  action {action:2d}: reward={reward:+.4f} p_beta_norm={info['p_beta_norm']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    rewards = [r for _, r, _ in rows]
    ranked = sorted(rows, key=lambda x: x[1], reverse=True)
    pattern0_rank = next(i + 1 for i, row in enumerate(ranked) if row[0] == 0)
    by_action = sorted(rows, key=lambda x: x[0])
    adj = [abs(by_action[i + 1][1] - by_action[i][1]) for i in range(len(by_action) - 1)]
    return {
        "observation_scale": scale,
        "reward_span": max(rewards) - min(rewards),
        "pattern0_rank": pattern0_rank,
        "best_action": ranked[0][0],
        "best_reward": ranked[0][1],
        "mean_adjacent_delta": sum(adj) / len(adj) if adj else 0.0,
    }


def merge_scales(scales: tuple[float, ...]) -> dict:
    merged: dict[str, object] = {
        "task": "TASK-164",
        "parent": "TASK-160",
        "mean_hz": MEAN_HZ,
        "state_length": STATE_LENGTH,
        "plant_dt_ms": PAPER_DT_MS,
        "seed": SEED,
        "scales": {},
    }
    missing: list[float] = []
    for scale in scales:
        path = _scale_path(scale)
        if not path.is_file():
            missing.append(scale)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged["scales"][str(scale)] = payload

    if missing:
        msg = f"missing partial artifacts for scales: {missing}"
        raise FileNotFoundError(msg)

    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_OUT.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=float,
        nargs="*",
        help="Run only these observation_scale values (writes partial JSON per scale)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge partial scale JSON files into task160_obs_scale_sweep.json",
    )
    args = parser.parse_args()

    if args.merge:
        scales = tuple(args.scale) if args.scale else DEFAULT_SCALES
        merged = merge_scales(scales)
        print(json.dumps(merged, indent=2), flush=True)
        print(f"wrote {FINAL_OUT}", file=sys.stderr)
        return 0

    scales = tuple(args.scale) if args.scale else DEFAULT_SCALES
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for scale in scales:
        print(f"=== scale={scale} ===", flush=True)
        row = run_scale(scale)
        out = _scale_path(scale)
        out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)

    if scales == DEFAULT_SCALES and len(scales) == len(DEFAULT_SCALES):
        merge_scales(DEFAULT_SCALES)
        print(f"wrote {FINAL_OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
