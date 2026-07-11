#!/usr/bin/env python3
"""TASK-108: Fig 5-style paper-protocol eval for pattern-mode policies.

Protocol: 2 s baseline (env.reset) + 5 repeated 2 s stimulation steps (§IV.A.2).
Compares constant pattern 0, best open-loop irregular from landscape JSON, and
optional trained DDPG checkpoints.

Run:
  uv run python scripts/run_task108_paper_protocol_eval.py --mean-hz 45 \\
    --landscape artifacts/ddpg/pattern_reward_landscape_45hz.json \\
    --checkpoint artifacts/ddpg/pattern_train_v2.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.ddpg import evaluate, load_actor
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.eval import EvalConfig, run_mehregan_eval
from controllers.ddpg.networks import Actor
from envs.mehregan.baselines import run_baseline_mehregan_eval
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.pattern_reward_landscape import describe_pattern


def _make_pattern_env(*, mean_hz: float, seed: int | None = None) -> MehreganEnv:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
        max_episode_steps=5,
    )
    plant = PythonPlant(config=resolved.plant)
    return MehreganEnv(plant=plant, config=env_cfg)


def _constant_action_eval(
    env: MehreganEnv,
    action: int,
    *,
    seed: int,
    label: str,
    eval_steps: int = 5,
) -> dict[str, Any]:
    _, info = env.reset(seed=seed)
    total_reward = float(info.get("reward", 0.0))
    rewards = [total_reward]
    p_beta = [float(info.get("p_beta_raw", np.nan))]
    stim = [float(info.get("dbs_freq_hz", 0.0))]

    for _ in range(eval_steps):
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        p_beta.append(float(info.get("p_beta_raw", np.nan)))
        stim.append(float(info.get("dbs_freq_hz", 0.0)))
        if terminated or truncated:
            break

    return {
        "label": label,
        "policy": "constant_action",
        "action": action,
        "seed": seed,
        "total_reward": total_reward,
        "reward_sum": total_reward,
        "rewards": rewards,
        "p_beta": p_beta,
        "stim_freq_hz": stim,
        "steps": len(rewards) - 1,
        "p_beta_mean": float(np.mean(p_beta)),
        "p_beta_final": float(p_beta[-1]),
        "stim_frequency_mean": float(np.mean(stim)),
        "protocol": "mehregan_eval",
        "eval_steps": eval_steps,
    }


def _no_stim_probe(*, seed: int) -> dict[str, Any]:
    """Single 2 s segment with no stimulation (scalar-frequency action 0)."""
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=1, action_space_mode="scalar_frequency")
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        env.reset(seed=seed)
        _obs, reward, _term, _trunc, info = env.step(0)
        return {
            "label": "no_stimulation",
            "reward": float(reward),
            "p_beta_raw": float(info["p_beta_raw"]),
            "p_beta_norm": float(info["p_beta_norm"]),
        }
    finally:
        env.close()


def _load_trainer_actor(
    path: Path,
    *,
    mean_hz: float,
    state_length: int = 1,
    n_actions: int = 41,
) -> Actor:
    cfg = DDPGConfig(action_space_mode="fixed_mean_pattern", pattern_mean_hz=mean_hz)
    cfg = cfg.with_variant_defaults()
    actor = Actor(
        state_length=state_length,
        n_actions=n_actions,
        conv1_out=cfg.conv1_out,
        conv2_out=cfg.conv2_out,
        pool_kernel=cfg.pool_kernel,
        fc_hidden=cfg.fc_hidden,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        msg = f"unrecognized checkpoint format: {path}"
        raise KeyError(msg)
    actor.eval()
    return actor


def _landscape_summary(landscape_path: Path) -> dict[str, Any]:
    payload = json.loads(landscape_path.read_text())
    summary = payload["summary"]
    best_action = int(summary["best_reward_action"])
    patterns = {int(p["action"]): p for p in payload["patterns"]}
    p0 = patterns[0]
    best = patterns[best_action]
    return {
        "mean_hz": float(payload["mean_hz"]),
        "seed": int(payload["seed"]),
        "pattern0_reward": float(p0["reward"]),
        "pattern0_p_beta_raw": float(p0["p_beta_raw"]),
        "pattern0_reward_rank": int(summary["pattern0_reward_rank"]),
        "best_reward_action": best_action,
        "best_reward": float(best["reward"]),
        "best_p_beta_raw": float(best["p_beta_raw"]),
        "best_semantics": describe_pattern(best_action, mean_hz=float(payload["mean_hz"])),
        "irregular_beat_pattern0_on_reward": bool(summary["irregular_beat_pattern0_on_reward"]),
    }


def run_eval(
    *,
    mean_hz: float,
    landscape_path: Path,
    checkpoints: list[Path],
    seed: int,
    eval_steps: int,
) -> dict[str, Any]:
    landscape = _landscape_summary(landscape_path)
    best_irregular = landscape["best_reward_action"]
    if best_irregular == 0:
        # Pick highest-reward non-zero pattern for the irregular baseline slot.
        payload = json.loads(landscape_path.read_text())
        candidates = [
            p for p in payload["patterns"] if int(p["action"]) != 0
        ]
        best_irregular = int(max(candidates, key=lambda p: p["reward"])["action"])

    env = _make_pattern_env(mean_hz=mean_hz)
    try:
        policies: dict[str, Any] = {
            "pattern0_regular": _constant_action_eval(
                env,
                0,
                seed=seed,
                label="pattern0_regular",
                eval_steps=eval_steps,
            ),
            "best_open_loop_irregular": _constant_action_eval(
                env,
                best_irregular,
                seed=seed,
                label=f"best_open_loop_irregular_a{best_irregular}",
                eval_steps=eval_steps,
            ),
        }

        for ckpt_path in checkpoints:
            if not ckpt_path.exists():
                policies[f"missing_{ckpt_path.stem}"] = {
                    "error": f"checkpoint not found: {ckpt_path}",
                }
                continue
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "actor_state_dict" in ckpt:
                    actor, _cfg = load_actor(ckpt_path)
                else:
                    actor = _load_trainer_actor(
                        ckpt_path, mean_hz=mean_hz, n_actions=env.action_space.n
                    )
                payload = run_mehregan_eval(
                    env,
                    actor,
                    config=EvalConfig(seed=seed, eval_steps=eval_steps),
                )
                payload["label"] = f"trained_ddpg_{ckpt_path.stem}"
                payload["checkpoint"] = str(ckpt_path)
                policies[payload["label"]] = payload
            except Exception as exc:  # noqa: BLE001 — report eval failures in artifact
                policies[f"error_{ckpt_path.stem}"] = {
                    "checkpoint": str(ckpt_path),
                    "error": repr(exc),
                }

        p0_total = policies["pattern0_regular"]["total_reward"]
        beats_p0 = {
            name: data["total_reward"]
            for name, data in policies.items()
            if isinstance(data, dict)
            and "total_reward" in data
            and name != "pattern0_regular"
            and float(data["total_reward"]) > p0_total
        }
    finally:
        env.close()

    no_stim = _no_stim_probe(seed=seed)

    return {
        "task": "TASK-108",
        "mean_hz": mean_hz,
        "seed": seed,
        "eval_steps": eval_steps,
        "landscape_1step": landscape,
        "no_stim_1step": no_stim,
        "paper_protocol_policies": policies,
        "pattern0_total_reward": p0_total,
        "any_beats_pattern0_on_total_reward": beats_p0,
        "pattern0_is_best_paper_protocol": len(beats_p0) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-hz", type=float, required=True)
    parser.add_argument("--landscape", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Trained DDPG checkpoint (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: artifacts/ddpg/task108_paper_protocol_{mean}hz.json)",
    )
    args = parser.parse_args()

    out = args.out or Path(
        f"artifacts/ddpg/task108_paper_protocol_{int(args.mean_hz)}hz.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    payload = run_eval(
        mean_hz=args.mean_hz,
        landscape_path=args.landscape,
        checkpoints=list(args.checkpoint),
        seed=args.seed,
        eval_steps=args.eval_steps,
    )
    payload["elapsed_s"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(
        {
            "out": str(out),
            "landscape_irregular_beats_p0": payload["landscape_1step"][
                "irregular_beat_pattern0_on_reward"
            ],
            "paper_protocol_beats_p0": list(
                payload["any_beats_pattern0_on_total_reward"].keys()
            ),
            "pattern0_is_best": payload["pattern0_is_best_paper_protocol"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
