#!/usr/bin/env python3
"""TASK-156: Q6 landscape — pattern 0 vs irregulars at 30 and 45 Hz mean rate.

Runs paper-aligned probes (plant.dt_ms=0.02) and writes:
  artifacts/ddpg/q6_landscape_1step_{30,45}hz.json
  artifacts/ddpg/q6_landscape_30step_{30,45}hz.json

No DDPG training — landscape / constant-policy rollouts only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.lib.pattern_reward_landscape import describe_pattern, run_landscape

PAPER_DT_MS = 0.02
ARTIFACTS = Path("artifacts/ddpg")
PROBE_SEED = 0
ROLLOUT_SEED = 1000
MAX_STEPS = 30
MEAN_HZ_VALUES = (30.0, 45.0)


def _make_env(*, mean_hz: float, max_episode_steps: int) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
        max_episode_steps=max_episode_steps,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg)


def _rollout_constant(env: MehreganEnv, *, seed: int, action: int) -> dict:
    env.reset(seed=seed)
    total_reward = 0.0
    step_rewards: list[float] = []
    p_beta_norm: list[float] = []
    for _ in range(MAX_STEPS):
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_rewards.append(float(reward))
        p_beta_norm.append(float(info.get("p_beta_norm", np.nan)))
        if terminated or truncated:
            break
    return {
        "action": action,
        "semantics": describe_pattern(action, mean_hz=env.config.pattern_mean_hz),
        "total_reward": total_reward,
        "mean_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "final_p_beta_norm": p_beta_norm[-1] if p_beta_norm else np.nan,
        "step_rewards": step_rewards,
        "p_beta_norm_trace": p_beta_norm,
    }


def run_30step_rollouts(
    env: MehreganEnv,
    *,
    seed: int,
    best_irregular_action: int,
    mid_irregular_action: int = 10,
    worst_action: int = 40,
) -> dict:
    rng = np.random.default_rng(seed + 7)
    policies: dict[str, int | str] = {
        "pattern_0_regular": 0,
        "best_irregular_1step": best_irregular_action,
        "mid_irregular": mid_irregular_action,
        "max_jitter_worst_1step": worst_action,
    }
    results: dict[str, dict] = {}
    for name, action in policies.items():
        if not isinstance(action, int):
            continue
        print(f"  30-step constant action {action} ({name})", flush=True)
        results[name] = _rollout_constant(env, seed=seed, action=action)

    # Random irregular rotation: new random irregular each step (actions 1–40).
    print("  30-step random irregular rotation", flush=True)
    env.reset(seed=seed)
    total = 0.0
    step_rewards: list[float] = []
    actions_taken: list[int] = []
    for _ in range(MAX_STEPS):
        action = int(rng.integers(1, 41))
        actions_taken.append(action)
        _obs, reward, terminated, truncated, _info = env.step(action)
        total += float(reward)
        step_rewards.append(float(reward))
        if terminated or truncated:
            break
    results["random_irregular_rotation"] = {
        "action": "random_1_40_per_step",
        "semantics": "uniform random irregular (1–40) each step",
        "total_reward": total,
        "mean_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "actions": actions_taken,
        "step_rewards": step_rewards,
    }

    p0_total = results["pattern_0_regular"]["total_reward"]
    beats_p0 = {
        name: data["total_reward"]
        for name, data in results.items()
        if name != "pattern_0_regular" and data["total_reward"] > p0_total
    }
    ranked = sorted(
        ((name, data["total_reward"]) for name, data in results.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "task": "TASK-156",
        "probe": "30step_constant_policy",
        "seed": seed,
        "mean_hz": env.config.pattern_mean_hz,
        "plant_dt_ms": PAPER_DT_MS,
        "max_steps": MAX_STEPS,
        "policies": results,
        "ranking_by_total_reward": [{"name": n, "total_reward": v} for n, v in ranked],
        "pattern0_total_reward": p0_total,
        "beaters_of_pattern0": beats_p0,
        "pattern0_is_best": len(beats_p0) == 0,
        "irregular_beat_pattern0_30step": len(beats_p0) > 0,
    }


def _worst_reward_action(patterns: list[dict]) -> int:
    return int(min(patterns, key=lambda p: p["reward"])["action"])


def run_mean_hz(mean_hz: float) -> dict[str, Path]:
    tag = f"{int(mean_hz)}hz"
    out_1step = ARTIFACTS / f"q6_landscape_1step_{tag}.json"
    out_30step = ARTIFACTS / f"q6_landscape_30step_{tag}.json"

    print(f"\n=== Q6 1-step sweep @ {mean_hz} Hz (dt={PAPER_DT_MS} ms) ===", flush=True)
    t0 = time.time()
    payload_1 = run_landscape(
        seed=PROBE_SEED,
        mean_hz=mean_hz,
        state_length=1,
        plant_dt_ms=PAPER_DT_MS,
    )
    payload_1["task"] = "TASK-156"
    payload_1["plant_dt_ms"] = PAPER_DT_MS
    payload_1["elapsed_s"] = round(time.time() - t0, 2)
    out_1step.parent.mkdir(parents=True, exist_ok=True)
    out_1step.write_text(json.dumps(payload_1, indent=2) + "\n")
    print(f"Wrote {out_1step}", flush=True)

    patterns = payload_1["patterns"]
    best_irregular = int(payload_1["summary"]["best_reward_action"])
    if best_irregular == 0:
        irregulars = [p for p in patterns if p["action"] != 0]
        best_irregular = int(max(irregulars, key=lambda p: p["reward"])["action"])
    worst = _worst_reward_action(patterns)

    print(f"\n=== Q6 30-step rollouts @ {mean_hz} Hz ===", flush=True)
    t1 = time.time()
    env = _make_env(mean_hz=mean_hz, max_episode_steps=MAX_STEPS)
    try:
        payload_30 = run_30step_rollouts(
            env,
            seed=ROLLOUT_SEED,
            best_irregular_action=best_irregular,
            worst_action=worst,
        )
    finally:
        env.close()
    payload_30["elapsed_s"] = round(time.time() - t1, 2)
    payload_30["best_irregular_1step_action"] = best_irregular
    payload_30["worst_1step_action"] = worst
    out_30step.write_text(json.dumps(payload_30, indent=2) + "\n")
    print(f"Wrote {out_30step}", flush=True)

    return {"1step": out_1step, "30step": out_30step}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-hz",
        type=float,
        nargs="+",
        default=list(MEAN_HZ_VALUES),
        help="Mean rates to probe (default: 30 45)",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    manifest: dict = {
        "task": "TASK-156",
        "plant_dt_ms": PAPER_DT_MS,
        "probe_seed": PROBE_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "mean_hz_runs": [],
    }

    for mean_hz in args.mean_hz:
        paths = run_mean_hz(mean_hz)
        manifest["mean_hz_runs"].append({"mean_hz": mean_hz, "outputs": {k: str(v) for k, v in paths.items()}})

    manifest["elapsed_s"] = round(time.time() - t_all, 2)
    manifest_path = ARTIFACTS / "q6_landscape_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone in {manifest['elapsed_s']}s — manifest {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    from scripts.lib.train_runtime_guard import run_main

    sys.exit(run_main(main, label="q6-landscape"))
