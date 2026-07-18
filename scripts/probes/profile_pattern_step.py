#!/usr/bin/env python3
"""Profile pattern-mode env.step vs scalar_frequency (TASK-102).

Run:
    uv run python scripts/probes/profile_pattern_step.py
    uv run python scripts/probes/profile_pattern_step.py --with-ddpg-buffer  # include 32-step replay fill

Writes a JSON summary to ``artifacts/profile_pattern_step.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.reward import mehregan_reward
from envs.plant.dbs import create_dbs_current
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACT = Path("artifacts/profile_pattern_step.json")


@dataclass
class StepBreakdown:
    to_dbs_spec_ms: float
    integrate_s: float
    reward_obs_ms: float
    total_s: float


@dataclass
class ModeProfile:
    mode: str
    n_actions: int
    reset_s_mean: float
    step_s_mean: float
    step_s_min: float
    step_s_max: float
    breakdown: StepBreakdown
    episode_s_est: float


def _mean_min_max(values: list[float]) -> tuple[float, float, float]:
    return stats.mean(values), min(values), max(values)


def _warmup_env(env: MehreganEnv, *, seed: int = 0, steps: int = 3) -> None:
    env.reset(seed=seed)
    for i in range(steps):
        env.step(i % env.action_space.n)


def _time_resets(env: MehreganEnv, *, n: int = 3, seed_base: int = 100) -> list[float]:
    times: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        env.reset(seed=seed_base + i)
        times.append(time.perf_counter() - t0)
    return times


def _time_steps(env: MehreganEnv, *, n: int = 5, seed: int = 0) -> list[float]:
    times: list[float] = []
    state, _ = env.reset(seed=seed)
    for i in range(n):
        action = (i * 7 + 3) % env.action_space.n
        t0 = time.perf_counter()
        state, _, terminated, truncated, _ = env.step(action)
        times.append(time.perf_counter() - t0)
        if terminated or truncated:
            state, _ = env.reset(seed=seed + i + 1)
    return times


def _breakdown_steps(env: MehreganEnv, *, n: int = 5, seed: int = 1) -> list[StepBreakdown]:
    cfg = env.config
    rows: list[StepBreakdown] = []
    env.reset(seed=seed)
    for i in range(n):
        action = (i * 7) % env.action_space.n
        t0 = time.perf_counter()
        spec = env.alphabet.to_dbs_spec(action)
        t1 = time.perf_counter()
        result = env._integrate_segment(spec)
        t2 = time.perf_counter()
        obs = env._push_observation(result.p_beta)
        _ = mehregan_reward(
            obs,
            beta_threshold=cfg.beta_threshold,
            reward_scale=cfg.reward_scale,
        )
        t3 = time.perf_counter()
        rows.append(
            StepBreakdown(
                to_dbs_spec_ms=(t1 - t0) * 1000.0,
                integrate_s=t2 - t1,
                reward_obs_ms=(t3 - t2) * 1000.0,
                total_s=t3 - t0,
            )
        )
    return rows


def _profile_mode(
    *,
    mode: str,
    resolved,
    max_episode_steps: int,
    n_step_samples: int,
) -> ModeProfile:
    cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode=mode,
        pattern_mean_hz=45.0,
    )
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=cfg)
    _warmup_env(env)
    reset_times = _time_resets(env)
    step_times = _time_steps(env, n=n_step_samples)
    breakdown_rows = _breakdown_steps(env, n=n_step_samples)
    n_actions = env.action_space.n
    env.close()

    reset_mean, _, _ = _mean_min_max(reset_times)
    step_mean, step_min, step_max = _mean_min_max(step_times)
    bd = StepBreakdown(
        to_dbs_spec_ms=stats.mean(r.to_dbs_spec_ms for r in breakdown_rows),
        integrate_s=stats.mean(r.integrate_s for r in breakdown_rows),
        reward_obs_ms=stats.mean(r.reward_obs_ms for r in breakdown_rows),
        total_s=stats.mean(r.total_s for r in breakdown_rows),
    )
    return ModeProfile(
        mode=mode,
        n_actions=n_actions,
        reset_s_mean=reset_mean,
        step_s_mean=step_mean,
        step_s_min=step_min,
        step_s_max=step_max,
        breakdown=bd,
        episode_s_est=step_mean * max_episode_steps,
    )


def _idbs_cache_profile() -> dict[str, float]:
    alphabet = FixedMeanPatternAlphabet(mean_hz=45.0)
    from envs.mehregan import fixed_mean_patterns as fmp

    fmp._build_idbs.cache_clear()
    t0 = time.perf_counter()
    for i in range(alphabet.n_actions):
        alphabet.idbs_for_pattern(i)
    cold_all_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(200):
        alphabet.idbs_for_pattern(19)
    warm_us = (time.perf_counter() - t0) / 200.0 * 1e6

    t0 = time.perf_counter()
    for _ in range(200):
        create_dbs_current(45.0, tmax_ms=2000.0, dt_ms=0.01)
    scalar_gen_ms = (time.perf_counter() - t0) / 200.0 * 1000.0

    return {
        "idbs_cold_all_41_patterns_ms": cold_all_ms,
        "idbs_warm_lookup_us": warm_us,
        "create_dbs_current_ms": scalar_gen_ms,
    }


def _ddpg_step_profile(resolved, *, buffer_fill: int) -> dict[str, float]:
    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
    )
    env = MehreganEnv(plant=PythonPlant(config=resolved.plant), config=env_cfg)
    config = DDPGConfig(
        variant="paper",
        seed=0,
        num_episodes=1,
        exploration_mode="softmax",
        exploration_temperature_start=1.0,
        exploration_temperature_end=1.0,
        critic_action_input="one_hot",
        critic_warmup_steps=0,
        reward_normalize=False,
        critic_loss_fn="mse",
        logit_noise_std=0.0,
        log_episodes=True,
        random_warmup_steps=0,
        batch_size=32,
        min_buffer_size=32,
        update_frequency=1,
    )
    trainer = DDPGTrainer(env, config)
    _warmup_env(env, steps=2)

    state, _ = env.reset(seed=0)
    if buffer_fill > 0:
        print(f"  Filling replay buffer ({buffer_fill} random steps)...", flush=True)
        for i in range(buffer_fill):
            action = int(env.action_space.sample())
            next_state, reward, terminated, truncated, info = env.step(action)
            trainer.buffer.add(
                state=state,
                action=action,
                action_logits=np.zeros(env.action_space.n, dtype=np.float32),
                reward=reward,
                next_state=next_state,
                dw=float(info.get("dw", 0.0)),
            )
            state = next_state
            if terminated or truncated:
                state, _ = env.reset(seed=i + 100)
    else:
        # Synthetic seed so _update_step can run without a long env warmup.
        dummy = np.zeros(env.observation_space.shape, dtype=np.float32)
        for _ in range(config.batch_size):
            trainer.buffer.add(
                state=dummy,
                action=0,
                action_logits=np.zeros(env.action_space.n, dtype=np.float32),
                reward=0.0,
                next_state=dummy,
                dw=0.0,
            )

    env_times: list[float] = []
    update_times: list[float] = []
    state, _ = env.reset(seed=200)
    for i in range(5):
        t_env0 = time.perf_counter()
        action, logits = trainer._select_action(state, env_step=buffer_fill + i)
        next_state, reward, terminated, truncated, info = env.step(action)
        env_times.append(time.perf_counter() - t_env0)

        t_up0 = time.perf_counter()
        trainer.buffer.add(
            state=state,
            action=action,
            action_logits=logits,
            reward=float(reward),
            next_state=next_state,
            dw=float(info.get("dw", 0.0)),
        )
        trainer._update_step()
        update_times.append(time.perf_counter() - t_up0)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset(seed=300 + i)

    env.close()
    return {
        "env_step_s_mean": stats.mean(env_times),
        "ddpg_update_s_mean": stats.mean(update_times),
        "total_train_step_s_mean": stats.mean(a + b for a, b in zip(env_times, update_times)),
    }


def _print_table(pattern: ModeProfile, scalar: ModeProfile, *, max_episode_steps: int) -> None:
    print("\n=== Timing summary (post-warmup) ===", flush=True)
    print(f"{'metric':<28} {'pattern':>12} {'scalar':>12}", flush=True)
    print("-" * 54, flush=True)
    rows = [
        ("reset (mean)", pattern.reset_s_mean, scalar.reset_s_mean),
        ("step (mean)", pattern.step_s_mean, scalar.step_s_mean),
        ("step (min)", pattern.step_s_min, scalar.step_s_min),
        ("step (max)", pattern.step_s_max, scalar.step_s_max),
        ("integrate (mean)", pattern.breakdown.integrate_s, scalar.breakdown.integrate_s),
        ("to_dbs_spec (mean ms)", pattern.breakdown.to_dbs_spec_ms, scalar.breakdown.to_dbs_spec_ms),
        (f"episode est ({max_episode_steps} steps)", pattern.episode_s_est, scalar.episode_s_est),
    ]
    for label, pval, sval in rows:
        if "ms" in label:
            print(f"{label:<28} {pval:>11.3f} {sval:>11.3f}", flush=True)
        else:
            print(f"{label:<28} {pval:>11.2f}s {sval:>11.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-ddpg-buffer",
        action="store_true",
        help="Fill 32-step replay buffer before DDPG timing (adds ~8+ min)",
    )
    parser.add_argument("--step-samples", type=int, default=5)
    args = parser.parse_args()

    resolved = resolve_config()
    max_episode_steps = MehreganEnvConfig().max_episode_steps

    print("=== Pattern-mode env.step profiler (TASK-102) ===", flush=True)
    print(f"plant: PythonPlant  step_duration=2s  max_episode_steps={max_episode_steps}", flush=True)

    print("\nProfiling fixed_mean_pattern...", flush=True)
    pattern = _profile_mode(
        mode="fixed_mean_pattern",
        resolved=resolved,
        max_episode_steps=max_episode_steps,
        n_step_samples=args.step_samples,
    )
    print(f"  n_actions={pattern.n_actions}", flush=True)

    print("\nProfiling scalar_frequency...", flush=True)
    scalar = _profile_mode(
        mode="scalar_frequency",
        resolved=resolved,
        max_episode_steps=max_episode_steps,
        n_step_samples=args.step_samples,
    )

    cache = _idbs_cache_profile()
    print("\n--- Pattern / idbs overhead ---", flush=True)
    print(f"  cold build all 41 patterns: {cache['idbs_cold_all_41_patterns_ms']:.2f} ms", flush=True)
    print(f"  warm idbs lookup:            {cache['idbs_warm_lookup_us']:.1f} µs", flush=True)
    print(f"  create_dbs_current (scalar): {cache['create_dbs_current_ms']:.3f} ms", flush=True)

    print("\n--- DDPG training step (pattern mode) ---", flush=True)
    ddpg = _ddpg_step_profile(resolved, buffer_fill=32 if args.with_ddpg_buffer else 0)
    print(
        f"  env.step={ddpg['env_step_s_mean']:.2f}s  "
        f"update={ddpg['ddpg_update_s_mean']:.2f}s  "
        f"total={ddpg['total_train_step_s_mean']:.2f}s",
        flush=True,
    )

    _print_table(pattern, scalar, max_episode_steps=max_episode_steps)

    print("\n--- Bottleneck ---", flush=True)
    pct = 100.0 * pattern.breakdown.integrate_s / pattern.breakdown.total_s
    print(
        f"  plant integration (numba CBGT loop): ~{pct:.1f}% of env.step; "
        "pattern idbs is lru_cached — not a bottleneck.",
        flush=True,
    )
    ratio = pattern.step_s_mean / scalar.step_s_mean if scalar.step_s_mean else 0.0
    print(
        f"  pattern vs scalar step ratio: {ratio:.2f}x "
        f"(irregular waveforms, not pattern-build overhead).",
        flush=True,
    )
    ep_20_h = pattern.episode_s_est * 20 / 3600.0
    print(f"  20-episode ETA (pattern): ~{ep_20_h:.1f} h", flush=True)

    payload = {
        "pattern": asdict(pattern),
        "scalar": asdict(scalar),
        "idbs_cache": cache,
        "ddpg": ddpg,
        "max_episode_steps": max_episode_steps,
        "bottleneck": "plant_integration",
        "optimization_status": "idbs_lru_cache_already_present",
    }
    def _json_default(obj: object) -> object:
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {ARTIFACT}", flush=True)
    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
