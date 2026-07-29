#!/usr/bin/env python3
"""Paper-path ablation: within_step L=16 + full_segment reward, burst skip_regular @ 45 Hz.

Tests whether temporal CNN input + continuous plant integration breaks constant-policy lock.

  tmux new-session -d -s within-step-l16-continuous \\
    \"setsid nohup uv run python -m rl_adaptive_dbs.run \\
      scripts/probes/alphabet_diversity/run_within_step_L16_burst_train.py --plant-integration continuous \\
      >> logs/within-step-l16-continuous.log 2>&1 < /dev/null\"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from controllers.ddpg import load_actor, train
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.quantization import prepare_actor_for_eval
from envs.mehregan.extensions.alphabet_diversity.config import WithinStepEnvConfig
from envs.mehregan.extensions.alphabet_diversity.env import WithinStepMehreganEnv
from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

MEAN_HZ = 45.0
SEED = 0
STATE_LENGTH = 16
STEP_DURATION_S = 2.0  # paper §IV.A.1
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30
PAPER_DT_MS = 0.02
INIT_BIAS_SCALE = 0.5
ROLLOUT_STEPS = 24

ARTIFACTS = Path("artifacts/ddpg")
CHECKPOINT_DISCONNECTED = ARTIFACTS / "checkpoint_within_step_L16_burst_skip_regular_2s.pt"
OUT_JSON_DISCONNECTED = ARTIFACTS / "within_step_L16_burst_train.json"
CHECKPOINT_CONTINUOUS = (
    ARTIFACTS / "checkpoint_within_step_L16_burst_skip_regular_2s_continuous.pt"
)
OUT_JSON_CONTINUOUS = ARTIFACTS / "within_step_L16_burst_train_continuous.json"


@torch.no_grad()
def _post_train_diagnostics(
    *,
    checkpoint: Path,
    env: WithinStepMehreganEnv,
    n_rollout: int,
) -> dict[str, object]:
    actor, _cfg = load_actor(checkpoint)
    actor.eval()
    n_actions = int(env.action_space.n)

    margins: list[float] = []
    greedy_actions: list[int] = []
    states: list[np.ndarray] = []

    obs, _ = env.reset(seed=SEED)
    for i in range(n_rollout):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = actor(x).squeeze(0).detach().cpu().numpy()
        order = np.argsort(-logits)
        margins.append(float(logits[order[0]] - logits[order[1]]))
        greedy_actions.append(int(order[0]))
        states.append(np.asarray(obs, dtype=np.float32).copy())
        action = int(order[0]) if i % 3 else int(order[min(3, len(order) - 1)])
        obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=SEED + i + 1)

    ptq_flips = 0
    fp_actor, _ = load_actor(checkpoint)
    ptq = prepare_actor_for_eval(
        fp_actor, "ptq-int8", device="cpu", ptq_weight_noise=0.0
    )
    ptq.eval()
    for st, a0 in zip(states, greedy_actions, strict=True):
        x = torch.as_tensor(st, dtype=torch.float32).unsqueeze(0)
        a_ptq = int(torch.argmax(ptq(x).squeeze(0)).item())
        if a_ptq != a0:
            ptq_flips += 1

    unique = sorted(set(greedy_actions))
    return {
        "n_rollout": n_rollout,
        "n_unique_greedy": len(unique),
        "unique_greedy_actions": unique,
        "constant_greedy_lock": len(unique) == 1,
        "mean_top1_top2_margin": float(np.mean(margins)),
        "median_top1_top2_margin": float(np.median(margins)),
        "max_top1_top2_margin": float(np.max(margins)),
        "ptq_noise0_argmax_flips": ptq_flips,
        "ptq_noise0_can_flip": ptq_flips > 0,
        "margins": margins,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plant-integration",
        choices=("disconnected", "continuous"),
        default="continuous",
        help="WithinStepMehreganEnv plant_integration_mode (default: continuous)",
    )
    args = parser.parse_args()
    plant_mode = args.plant_integration
    checkpoint = (
        CHECKPOINT_CONTINUOUS if plant_mode == "continuous" else CHECKPOINT_DISCONNECTED
    )
    out_json = (
        OUT_JSON_CONTINUOUS if plant_mode == "continuous" else OUT_JSON_DISCONNECTED
    )

    t0 = time.time()
    print(
        f"=== within_step L={STATE_LENGTH} burst skip_regular @ {MEAN_HZ:g} Hz ===",
        flush=True,
    )
    print(
        f"step={STEP_DURATION_S}s reward=full_segment plant={plant_mode} "
        f"eps={NUM_EPISODES} greedy+logits seed={SEED}",
        flush=True,
    )

    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = WithinStepEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        reward_state_mode="full_segment",
        step_duration_s=STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=True,
        plant_dt_ms=PAPER_DT_MS,
        plant_integration_mode=plant_mode,
    )
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=STEP_DURATION_S,
        dt_ms=PAPER_DT_MS,
        skip_regular=True,
    )
    env = WithinStepMehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,
    )
    print(
        f"obs shape={env.observation_space.shape} n_actions={alphabet.n_actions}",
        flush=True,
    )

    ddpg_cfg = fig4a_ddpg_config(
        seed=SEED,
        num_episodes=NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
        pattern_mean_hz=MEAN_HZ,
        exploration_mode="greedy",
        critic_action_input="logits",
        init_bias_scale=INIT_BIAS_SCALE,
        logit_noise_std=0.0,
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = train(config=ddpg_cfg, env=env, checkpoint_path=checkpoint)

    train_unique: list[int] = []
    if hasattr(result, "episode_actions") and result.episode_actions:
        all_actions = [a for ep in result.episode_actions for a in ep]
        train_unique = sorted(set(all_actions))
        print(
            f"train unique actions: {len(train_unique)} / {alphabet.n_actions}",
            flush=True,
        )

    diag = _post_train_diagnostics(
        checkpoint=checkpoint, env=env, n_rollout=ROLLOUT_STEPS
    )
    payload = {
        "mean_hz": MEAN_HZ,
        "state_mode": "within_step",
        "state_length": STATE_LENGTH,
        "reward_state_mode": "full_segment",
        "plant_integration_mode": plant_mode,
        "step_duration_s": STEP_DURATION_S,
        "num_episodes": NUM_EPISODES,
        "exploration": "greedy",
        "critic_action_input": "logits",
        "checkpoint": str(checkpoint),
        "train_unique_actions": train_unique,
        "n_train_unique": len(train_unique),
        "diagnostics": {k: v for k, v in diag.items() if k != "margins"},
        "margins_raw": diag["margins"],
        "elapsed_s": round(time.time() - t0, 2),
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["diagnostics"], indent=2), flush=True)
    print(f"wrote {checkpoint} and {out_json}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
