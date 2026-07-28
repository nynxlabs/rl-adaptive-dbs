#!/usr/bin/env python3
"""Cheap plant continuity probe: disconnected 2 s steps vs one stitched integrate.

Compares WithinStepMehreganEnv-style integration (each step cold-restarts from the same
IC draw) against Fig 5a-style continuous simulation (one long idbs timeline).

Run standalone:
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/probes/alphabet_diversity/run_plant_continuity_probe.py

Chained after within_step train:
  scripts/probes/alphabet_diversity/run_after_within_step_pipeline.sh
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant.biomarkers import p_beta
from envs.plant.dbs import DbsSpec
from envs.plant.python_backend import PythonPlant
from envs.mehregan.extensions.alphabet_diversity.observations import clip_spikes_to_subwindow
from rl_adaptive_dbs.user_config import resolve_config

MEAN_HZ = 45.0
SEED = 0
PAPER_DT_MS = 0.02
PRE_STIM_S = 2.0
STEP_S = 2.0
N_STIM_STEPS = 5
OUT_JSON = Path("artifacts/ddpg/plant_continuity_probe.json")

# Fig 5a / open-loop favourites for sequence probe
SEQUENCES: dict[str, list[int]] = {
    "constant_7": [7, 7, 7, 7, 7],
    "constant_28": [28, 28, 28, 28, 28],
    "mixed_top": [15, 35, 3, 23, 9],  # burst n=41 near-best @ 45 Hz
}


def _stitch_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int],
    alphabet: BurstPatternAlphabet,
    rl_step_s: float,
) -> np.ndarray:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    trace = np.zeros(n_steps, dtype=np.float64)
    onset_idx = int(round(onset_sim_s * 1000.0 / dt_ms))
    step_samples = int(round(rl_step_s * 1000.0 / dt_ms))
    for seg_i, action in enumerate(segment_actions):
        seg = np.asarray(alphabet.idbs_for_action(int(action)), dtype=np.float64)
        start = onset_idx + seg_i * step_samples
        end = min(start + seg.size, n_steps)
        if start >= n_steps:
            break
        trace[start:end] = seg[: end - start]
    return trace


def _p_beta_window(
    gpi_spikes: list[np.ndarray],
    *,
    t_start_s: float,
    t_end_s: float,
    dt_ms: float,
) -> float:
    window_spikes = clip_spikes_to_subwindow(
        gpi_spikes, t_start_s=t_start_s, t_end_s=t_end_s
    )
    sub_dur = t_end_s - t_start_s
    return float(
        p_beta(window_spikes, dt_ms=dt_ms, segment_duration_s=sub_dur)
    )


def _disconnected_sequence(
    plant: PythonPlant,
    *,
    seed: int,
    actions: list[int],
    alphabet: BurstPatternAlphabet,
) -> dict[str, Any]:
    """WithinStepMehreganEnv-like: reset + optional pre-stim, then cold 2 s steps."""
    dt_ms = float(plant.config.dt_ms)
    plant.reset(seed)
    pre = plant.integrate(PRE_STIM_S, DbsSpec.none())
    step_rows: list[dict[str, Any]] = []
    for i, action in enumerate(actions):
        spec = alphabet.to_dbs_spec(int(action))
        result = plant.integrate(STEP_S, spec)
        step_rows.append(
            {
                "step": i,
                "action": int(action),
                "p_beta_raw": float(result.p_beta),
            }
        )
    return {
        "pre_stim_p_beta": float(pre.p_beta),
        "steps": step_rows,
        "final_p_beta": step_rows[-1]["p_beta_raw"] if step_rows else None,
    }


def _continuous_sequence(
    plant: PythonPlant,
    *,
    seed: int,
    actions: list[int],
    alphabet: BurstPatternAlphabet,
) -> dict[str, Any]:
    """Fig 5a-like: one integrate with stitched idbs after pre-stim onset."""
    dt_ms = float(plant.config.dt_ms)
    total_s = PRE_STIM_S + len(actions) * STEP_S
    idbs = _stitch_idbs(
        duration_s=total_s,
        dt_ms=dt_ms,
        onset_sim_s=PRE_STIM_S,
        segment_actions=actions,
        alphabet=alphabet,
        rl_step_s=STEP_S,
    )
    spec = DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(MEAN_HZ).pick_dbs_freq,
        idbs=idbs,
        mean_hz=MEAN_HZ,
    )
    plant.reset(seed)
    result = plant.integrate(total_s, spec)
    if not result.gpi_spikes:
        msg = "continuous integrate missing gpi_spikes"
        raise RuntimeError(msg)

    pre_pb = _p_beta_window(
        result.gpi_spikes, t_start_s=0.0, t_end_s=PRE_STIM_S, dt_ms=dt_ms
    )
    step_rows: list[dict[str, Any]] = []
    for i, action in enumerate(actions):
        t0 = PRE_STIM_S + i * STEP_S
        t1 = t0 + STEP_S
        step_rows.append(
            {
                "step": i,
                "action": int(action),
                "p_beta_raw": _p_beta_window(
                    result.gpi_spikes, t_start_s=t0, t_end_s=t1, dt_ms=dt_ms
                ),
            }
        )
    return {
        "pre_stim_p_beta": pre_pb,
        "steps": step_rows,
        "final_p_beta": step_rows[-1]["p_beta_raw"] if step_rows else None,
        "whole_segment_p_beta": float(result.p_beta) if result.p_beta else None,
    }


def _open_loop_ranking(
    plant: PythonPlant,
    *,
    seed: int,
    alphabet: BurstPatternAlphabet,
) -> dict[str, Any]:
    """Per-pattern 2 s open-loop: disconnected vs continuous (2 s pre + 2 s stim)."""
    dt_ms = float(plant.config.dt_ms)
    n = alphabet.n_actions
    disc: list[float] = []
    cont_stim: list[float] = []
    for action in range(n):
        plant.reset(seed)
        d = plant.integrate(STEP_S, alphabet.to_dbs_spec(action))
        disc.append(float(d.p_beta))

        total_s = PRE_STIM_S + STEP_S
        idbs = _stitch_idbs(
            duration_s=total_s,
            dt_ms=dt_ms,
            onset_sim_s=PRE_STIM_S,
            segment_actions=[action],
            alphabet=alphabet,
            rl_step_s=STEP_S,
        )
        spec = DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(MEAN_HZ).pick_dbs_freq,
            idbs=idbs,
            mean_hz=MEAN_HZ,
        )
        plant.reset(seed)
        c = plant.integrate(total_s, spec)
        if not c.gpi_spikes:
            cont_stim.append(float("nan"))
        else:
            cont_stim.append(
                _p_beta_window(
                    c.gpi_spikes,
                    t_start_s=PRE_STIM_S,
                    t_end_s=PRE_STIM_S + STEP_S,
                    dt_ms=dt_ms,
                )
            )

    disc_arr = np.asarray(disc, dtype=np.float64)
    cont_arr = np.asarray(cont_stim, dtype=np.float64)
    valid = np.isfinite(disc_arr) & np.isfinite(cont_arr)
    if valid.sum() < 3:
        spearman = None
    else:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(disc_arr[valid], cont_arr[valid]).correlation)

    disc_best = int(np.argmin(disc_arr))
    cont_best = int(np.argmin(cont_arr))
    return {
        "n_actions": n,
        "disconnected_best_action": disc_best,
        "continuous_best_action": cont_best,
        "best_action_match": disc_best == cont_best,
        "spearman_rank_correlation": spearman,
        "mean_abs_p_beta_delta": float(np.mean(np.abs(disc_arr - cont_arr))),
        "max_abs_p_beta_delta": float(np.max(np.abs(disc_arr - cont_arr))),
    }


def main() -> int:
    t0 = time.time()
    print("=== plant continuity probe ===", flush=True)
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    plant = PythonPlant(config=plant_cfg)
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=STEP_S,
        dt_ms=PAPER_DT_MS,
        skip_regular=True,
    )
    print(
        f"burst skip_regular n={alphabet.n_actions} seed={SEED} "
        f"pre={PRE_STIM_S}s step={STEP_S}s",
        flush=True,
    )

    payload: dict[str, Any] = {
        "seed": SEED,
        "mean_hz": MEAN_HZ,
        "pre_stim_s": PRE_STIM_S,
        "step_s": STEP_S,
        "n_stim_steps": N_STIM_STEPS,
        "sequences": {},
        "open_loop": {},
    }

    for name, actions in SEQUENCES.items():
        use = actions[:N_STIM_STEPS]
        print(f"\n--- sequence {name}: {use} ---", flush=True)
        disc = _disconnected_sequence(plant, seed=SEED, actions=use, alphabet=alphabet)
        cont = _continuous_sequence(plant, seed=SEED, actions=use, alphabet=alphabet)
        final_delta = None
        if disc["final_p_beta"] is not None and cont["final_p_beta"] is not None:
            final_delta = float(cont["final_p_beta"] - disc["final_p_beta"])
        print(
            f"  disconnected final Pβ={disc['final_p_beta']:.1f} "
            f"continuous final Pβ={cont['final_p_beta']:.1f} "
            f"delta={final_delta:.1f}" if final_delta is not None else "",
            flush=True,
        )
        payload["sequences"][name] = {
            "actions": use,
            "disconnected": disc,
            "continuous": cont,
            "final_p_beta_delta": final_delta,
        }

    print("\n--- open-loop ranking (40 patterns) ---", flush=True)
    ranking = _open_loop_ranking(plant, seed=SEED, alphabet=alphabet)
    payload["open_loop"] = ranking
    print(
        f"  best disconnected={ranking['disconnected_best_action']} "
        f"continuous={ranking['continuous_best_action']} "
        f"match={ranking['best_action_match']} "
        f"spearman={ranking['spearman_rank_correlation']}",
        flush=True,
    )

    payload["elapsed_s"] = round(time.time() - t0, 2)
    payload["verdict"] = (
        "continuity_matters"
        if (
            not ranking.get("best_action_match", True)
            or (ranking.get("spearman_rank_correlation") or 1.0) < 0.95
            or any(
                abs(s.get("final_p_beta_delta") or 0.0) > 15.0
                for s in payload["sequences"].values()
            )
        )
        else "continuity_minor"
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nverdict={payload['verdict']} wrote {OUT_JSON}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
