#!/usr/bin/env python3
"""Sweep all 41 patterns (0-40) at 45 Hz mean, report P_beta per action."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataclasses import replace

from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from rl_adaptive_dbs.user_config import resolve_config

SEED = 42
MEAN_HZ = 45.0

def main():
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=0.02)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=1,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)

    results = []
    t0 = time.time()
    for a in range(0, 41):
        env.reset(seed=SEED)
        obs, reward, term, trunc, info = env.step(a)
        pb = info.get("P_beta", obs[0])
        results.append((a, pb, reward))
        elapsed = time.time() - t0
        print(f"[{elapsed:6.0f}s] action {a:2d}: P_beta = {pb:.6f}, reward = {reward:.6f}", flush=True)

    total = time.time() - t0
    print(f"\n=== SWEEP COMPLETE ({total:.0f}s) ===\n")

    # Sorted by P_beta ascending
    sorted_r = sorted(results, key=lambda x: x[1])
    print("  action   P_beta      reward")
    print("  ------   ------      ------")
    for a, p, r in sorted_r:
        tag = "  <-- REGULAR (baseline)" if a == 0 else ""
        print(f"  {a:6d}   {p:.6f}   {r:.6f}{tag}")

    reg = next(r for r in results if r[0] == 0)
    irr = [r for r in results if r[0] != 0]
    best = min(irr, key=lambda x: x[1])
    worst = max(irr, key=lambda x: x[1])

    print(f"\n=== ANALYSIS ===")
    print(f"Regular (pattern 0):    P_beta = {reg[1]:.6f}")
    print(f"Best irregular:         action {best[0]}, P_beta = {best[1]:.6f}  (delta = +{best[1]-reg[1]:.6f})")
    print(f"Worst irregular:        action {worst[0]}, P_beta = {worst[1]:.6f}")
    print(f"Irregular spread:       {worst[1]-best[1]:.6f}")
    print(f"All irregulars > regular? {all(r[1] > reg[1] for r in irr)}")
    print(f"Mean irregular P_beta:  {sum(r[1] for r in irr)/len(irr):.6f}")

if __name__ == "__main__":
    main()
