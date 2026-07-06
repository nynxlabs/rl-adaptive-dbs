#!/usr/bin/env python3
"""Offline DDPG diagnostics for policy collapse (TASK-71).

Probes actor checkpoints without plant integration. Optional ``--plant`` runs a
short reward-sensitivity sweep (slow on PythonPlant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from controllers.ddpg.checkpoint import load_actor
from controllers.ddpg.networks import Actor
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.reward import mehregan_reward
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config


def probe_actor(actor: Actor, state_length: int, *, n_samples: int = 500) -> dict[str, float | int]:
    """Synthetic biomarker windows in normalized operating range (0.25–0.65)."""
    actor.eval()
    logits_list: list[np.ndarray] = []
    enc_list: list[np.ndarray] = []
    actions: list[int] = []
    with torch.no_grad():
        for _ in range(n_samples):
            window = np.random.uniform(0.25, 0.65, size=state_length).astype(np.float32)
            state_t = torch.as_tensor(window).unsqueeze(0)
            enc = actor.encoder(state_t)
            logits = actor(state_t)
            logits_list.append(logits.squeeze(0).numpy())
            enc_list.append(enc.squeeze(0).numpy())
            actions.append(int(torch.argmax(logits).item()))
    logits_arr = np.stack(logits_list)
    enc_arr = np.stack(enc_list)
    sorted_logits = np.sort(logits_arr, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    n_actions = int(actor.head.out_features)
    dominant = int(np.bincount(actions, minlength=n_actions).argmax())
    return {
        "unique_argmax": len(set(actions)),
        "dominant_action": dominant,
        "logit_margin_mean": float(margins.mean()),
        "logit_margin_min": float(margins.min()),
        "encoder_feature_std": float(enc_arr.std()),
        "encoder_feature_range": float(enc_arr.max() - enc_arr.min()),
        "head_bias_max": float(actor.head.bias.max().item()),
        "head_bias_argmax_action": int(actor.head.bias.argmax().item()),
    }


def synthetic_reward_summary() -> dict[str, float]:
    s_vals = np.linspace(0.2, 0.6, 41)
    rewards = [mehregan_reward(np.array([s])) for s in s_vals]
    return {
        "max_reward": float(max(rewards)),
        "reward_at_0.30": float(mehregan_reward(np.array([0.30]))),
        "reward_at_0.35": float(mehregan_reward(np.array([0.35]))),
        "reward_at_0.50": float(mehregan_reward(np.array([0.50]))),
        "reward_range_0.2_0.6": float(max(rewards) - min(rewards)),
    }


def plant_reward_sensitivity(*, seed: int = 42, action_stride: int = 4) -> dict[str, float | int]:
    resolved = resolve_config()
    env = MehreganEnv(
        plant=PythonPlant(config=resolved.plant),
        config=MehreganEnvConfig(state_length=15),
    )
    rewards: list[float] = []
    obs_means: list[float] = []
    try:
        for action in range(0, env.action_space.n, action_stride):
            state, _info = env.reset(seed=seed)
            state, reward, _term, _trunc, _info = env.step(action)
            rewards.append(float(reward))
            obs_means.append(float(np.mean(state)))
    finally:
        env.close()
    return {
        "reward_min": float(min(rewards)),
        "reward_max": float(max(rewards)),
        "reward_std": float(np.std(rewards)),
        "obs_mean_std": float(np.std(obs_means)),
        "n_sampled_actions": len(rewards),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        default=[
            Path("artifacts/ddpg/paper_train0.pt"),
            Path("artifacts/ddpg/paper_train0_explore.pt"),
        ],
    )
    parser.add_argument("--state-length", type=int, default=15)
    parser.add_argument("--plant", action="store_true", help="Run plant reward sensitivity (slow)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ddpg/ddpg_diagnose.json"))
    args = parser.parse_args()

    out: dict[str, object] = {"synthetic_reward": synthetic_reward_summary(), "checkpoints": {}}
    if args.plant:
        out["plant_reward_sensitivity"] = plant_reward_sensitivity()

    for path in args.checkpoints:
        if not path.exists():
            print(f"skip missing checkpoint: {path}", file=sys.stderr)
            continue
        actor, _cfg = load_actor(path)
        sl = args.state_length if "explore" in path.name or args.state_length != 15 else 1
        if path.name == "paper_train0.pt":
            sl = 1
        out["checkpoints"][str(path)] = probe_actor(actor, sl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
