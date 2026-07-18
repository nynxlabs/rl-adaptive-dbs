#!/usr/bin/env python3
"""TASK-176: Pattern alphabet oracle — does any pattern beat no-stim at 30 Hz?

Runs paper-aligned probes (within_step L=16, reward_state_mode=full_segment):
  1. No-stim baseline (scalar 0 Hz, single 2 s segment)
  2. Single-step Pβ landscape at dt=0.02 and dt=0.01 (resolve dt discrepancy)
  3. Fig 5b oracle: constant action, 2 s reset + 5×2 s steps, all 41 patterns

Artifacts:
  artifacts/ddpg/task176_pattern_oracle_30hz.json
  artifacts/ddpg/task176_pattern_oracle_30hz.log  (when tee'd)

Run:
  uv run python scripts/probes/run_task176_pattern_oracle_probe.py
  tmux new-session -d -s task176-oracle \\
    'uv run python scripts/probes/run_task176_pattern_oracle_probe.py 2>&1 | tee artifacts/ddpg/task176_pattern_oracle_30hz.log'
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.lib.pattern_reward_landscape import describe_pattern, run_landscape

ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "task176_pattern_oracle_30hz.json"

PAPER_DT_MS = 0.02
DEFAULT_DT_MS = 0.01
MEAN_HZ = 30.0
STATE_LENGTH = 16
PROBE_SEED = 0
EVAL_STEPS = 5


def _no_stim_probe(*, seed: int, plant_cfg) -> dict[str, Any]:
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        action_space_mode="scalar_frequency",
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        env.reset(seed=seed)
        _obs, reward, _term, _trunc, info = env.step(0)
        return {
            "label": "no_stimulation",
            "reward": float(reward),
            "p_beta_raw": float(info["p_beta_raw"]),
            "p_beta_norm": float(info["p_beta_norm"]),
            "protocol": "single 2s segment, scalar action 0 (0 Hz)",
        }
    finally:
        env.close()


def _fig5b_constant_eval(
    env: MehreganEnv,
    *,
    action: int,
    seed: int,
) -> dict[str, Any]:
    """Paper Fig 5b protocol: 2 s reset + 5 stimulation steps (constant action)."""
    _, info = env.reset(seed=seed)
    p_beta = [float(info.get("p_beta_raw", np.nan))]
    for _ in range(EVAL_STEPS):
        _, _reward, terminated, truncated, info = env.step(action)
        p_beta.append(float(info.get("p_beta_raw", np.nan)))
        if terminated or truncated:
            break
    return {
        "action": action,
        "seed": seed,
        "p_beta": p_beta,
        "p_beta_mean": float(np.mean(p_beta)),
        "eval_steps": EVAL_STEPS,
        "protocol": "mehregan_eval (2s reset + 5×2s steps)",
    }


def _make_pattern_env(*, plant_cfg) -> MehreganEnv:
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        reward_state_mode="full_segment",
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg)


def _landscape_summary(
    payload: dict,
    *,
    no_stim_p_beta: float,
    label: str,
) -> dict[str, Any]:
    patterns = payload["patterns"]
    beat = [p for p in patterns if float(p["p_beta_raw"]) < no_stim_p_beta]
    best = min(patterns, key=lambda p: float(p["p_beta_raw"]))
    return {
        "label": label,
        "plant_dt_ms": payload["plant_dt_ms"],
        "state_length": payload["state_length"],
        "no_stim_p_beta": no_stim_p_beta,
        "best_action": int(best["action"]),
        "best_p_beta_raw": float(best["p_beta_raw"]),
        "n_beat_no_stim": len(beat),
        "n_patterns": len(patterns),
        "beaters": [
            {"action": int(p["action"]), "p_beta_raw": float(p["p_beta_raw"])}
            for p in sorted(beat, key=lambda x: float(x["p_beta_raw"]))
        ],
    }


def main() -> int:
    t0 = time.time()
    resolved = resolve_config()
    plant_paper = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    plant_default = replace(resolved.plant, dt_ms=DEFAULT_DT_MS)

    print("=== TASK-176: pattern alphabet oracle @ 30 Hz ===", flush=True)

    no_stim = _no_stim_probe(seed=PROBE_SEED, plant_cfg=plant_paper)
    no_stim_p = no_stim["p_beta_raw"]
    print(f"no-stim Pβ (dt={PAPER_DT_MS}): {no_stim_p:.2f}", flush=True)

    landscapes: dict[str, dict] = {}
    for dt, plant_cfg, tag in (
        (PAPER_DT_MS, plant_paper, "dt002"),
        (DEFAULT_DT_MS, plant_default, "dt001"),
    ):
        print(f"\n--- 1-step landscape @ dt={dt} ms, L={STATE_LENGTH} ---", flush=True)
        t1 = time.time()
        payload = run_landscape(
            seed=PROBE_SEED,
            mean_hz=MEAN_HZ,
            state_length=STATE_LENGTH,
            plant_dt_ms=dt,
            reward_state_mode="full_segment",
        )
        payload["elapsed_s"] = round(time.time() - t1, 2)
        landscapes[tag] = payload
        s = _landscape_summary(payload, no_stim_p_beta=no_stim_p, label=tag)
        print(
            f"  best action {s['best_action']}: Pβ={s['best_p_beta_raw']:.2f}; "
            f"beat no-stim: {s['n_beat_no_stim']}/{s['n_patterns']}",
            flush=True,
        )

    print(f"\n--- Fig5b oracle: 41 constant patterns @ dt={PAPER_DT_MS} ---", flush=True)
    env = _make_pattern_env(plant_cfg=plant_paper)
    fig5b_rows: list[dict[str, Any]] = []
    try:
        for action in range(41):
            t2 = time.time()
            row = _fig5b_constant_eval(env, action=action, seed=PROBE_SEED)
            row["semantics"] = describe_pattern(action, mean_hz=MEAN_HZ)
            row["elapsed_s"] = round(time.time() - t2, 2)
            # Attach 1-step Pβ from paper-dt landscape for combined table
            l16 = landscapes["dt002"]["patterns"][action]
            row["p_beta_raw_1step"] = float(l16["p_beta_raw"])
            row["beat_no_stim_1step"] = row["p_beta_raw_1step"] < no_stim_p
            row["beat_no_stim_fig5b"] = row["p_beta_mean"] < no_stim_p
            fig5b_rows.append(row)
            print(
                f"  action {action:2d}: 1-step={row['p_beta_raw_1step']:.1f} "
                f"fig5b_mean={row['p_beta_mean']:.1f} "
                f"({'BEAT' if row['beat_no_stim_fig5b'] else 'lose'})",
                flush=True,
            )
    finally:
        env.close()

    fig5b_beat = [r for r in fig5b_rows if r["beat_no_stim_fig5b"]]
    best_fig5b = min(fig5b_rows, key=lambda r: r["p_beta_mean"])

    combined_table = [
        {
            "action": r["action"],
            "semantics": r["semantics"],
            "p_beta_raw_1step": r["p_beta_raw_1step"],
            "p_beta_mean_fig5b": r["p_beta_mean"],
            "beat_no_stim_1step": r["beat_no_stim_1step"],
            "beat_no_stim_fig5b": r["beat_no_stim_fig5b"],
        }
        for r in fig5b_rows
    ]

    result: dict[str, Any] = {
        "task": "TASK-176",
        "mean_hz": MEAN_HZ,
        "state_mode": "within_step",
        "state_length": STATE_LENGTH,
        "reward_state_mode": "full_segment",
        "seed": PROBE_SEED,
        "no_stim": no_stim,
        "landscapes": {
            tag: _landscape_summary(
                landscapes[tag],
                no_stim_p_beta=no_stim_p,
                label=tag,
            )
            for tag in landscapes
        },
        "landscape_full": landscapes,
        "fig5b_oracle": {
            "plant_dt_ms": PAPER_DT_MS,
            "protocol": "mehregan_eval (2s reset + 5×2s steps), constant action",
            "n_beat_no_stim": len(fig5b_beat),
            "n_patterns": len(fig5b_rows),
            "best_action": int(best_fig5b["action"]),
            "best_p_beta_mean": float(best_fig5b["p_beta_mean"]),
            "best_p_beta_1step": float(best_fig5b["p_beta_raw_1step"]),
            "beaters": [
                {
                    "action": int(r["action"]),
                    "p_beta_mean": float(r["p_beta_mean"]),
                    "p_beta_raw_1step": float(r["p_beta_raw_1step"]),
                }
                for r in sorted(fig5b_beat, key=lambda x: x["p_beta_mean"])
            ],
            "patterns": fig5b_rows,
        },
        "combined_table": combined_table,
        "conclusions": {
            "dt_discrepancy_resolved": True,
            "at_dt_0_02_L16": {
                "single_step_beat_no_stim": landscapes["dt002"]["patterns"]
                and _landscape_summary(landscapes["dt002"], no_stim_p_beta=no_stim_p, label="")[
                    "n_beat_no_stim"
                ],
                "fig5b_beat_no_stim": len(fig5b_beat),
                "structural_blocker_fig5b_constant": len(fig5b_beat) == 0,
            },
            "at_dt_0_01_L16": {
                "single_step_beat_no_stim": _landscape_summary(
                    landscapes["dt001"], no_stim_p_beta=no_stim_p, label=""
                )["n_beat_no_stim"],
            },
        },
        "elapsed_s": round(time.time() - t0, 2),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(result["conclusions"], indent=2), flush=True)
    print(f"total elapsed={result['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
