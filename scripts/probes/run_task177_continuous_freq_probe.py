#!/usr/bin/env python3
"""TASK-177 exp 6: Continuous scalar-frequency landscape (0–200 Hz @ 1 Hz).

COO request: test whether continuous action space contains frequencies that
beat no-stim (Pβ < 503.4) at dt=0.02 — resolves paper tension between discrete
patterns and DDPG's continuous-action superiority claim.

Unlike Kumaravelu PatternAlphabet (41 actions, 5 Hz grid), this sweeps every
integer Hz via precomputed idbs traces (ContinuousFrequencyAlphabet).

Single-step only (no Fig5b). Optionally reports discrete 41-grid comparison.

Artifacts:
  artifacts/ddpg/task177_continuous_freq_probe.json
  artifacts/ddpg/task177_continuous_freq_probe.log

Ops / resume notes: wilzen3476/tasks/task177-continuous-freq-probe.md (gitignored).

Run:
    scripts/probes/run_task177_continuous_freq_probe.py --continuous-only --resume
  tmux new-session -d -s task177-continuous \\
    'taskset -c 0-2 uv run python -m rl_adaptive_dbs.run --max-threads 3 \\
      scripts/probes/run_task177_continuous_freq_probe.py --continuous-only --resume \\
      2>&1 | tee -a artifacts/ddpg/task177_continuous_freq_probe.log'
"""

from __future__ import annotations

from rl_adaptive_dbs.thread_limits import bootstrap_thread_limits

_BOOTSTRAP_MAX_THREADS = bootstrap_thread_limits()

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.pattern_alternatives import ContinuousFrequencyAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.thread_limits import add_max_threads_argument, format_thread_limit_banner
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "task177_continuous_freq_probe.json"

PAPER_DT_MS = 0.02
STATE_LENGTH = 16
PROBE_SEED = 0
NO_STIM_REF = 503.4


def _make_env(*, plant_cfg, alphabet) -> MehreganEnv:
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        reward_state_mode="full_segment",
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=30.0,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _row_from_step(
    env: MehreganEnv,
    alphabet,
    *,
    action: int,
    no_stim: float,
    label_freq: bool,
) -> dict[str, Any]:
    env.reset(seed=PROBE_SEED)
    _obs, reward, _term, _trunc, info = env.step(action)
    p_beta = float(info["p_beta_raw"])
    row: dict[str, Any] = {
        "action": action,
        "reward": float(reward),
        "p_beta_raw": p_beta,
        "beat_no_stim": p_beta < no_stim,
    }
    if label_freq and hasattr(alphabet, "frequency_hz_for_action"):
        row["frequency_hz"] = float(alphabet.frequency_hz_for_action(action))
    elif hasattr(alphabet, "to_dbs_spec"):
        row["frequency_hz"] = float(alphabet.to_dbs_spec(action).frequency_hz)
    return row


def _landscape(
    env: MehreganEnv,
    alphabet,
    *,
    no_stim: float,
    label_freq: bool = False,
    existing_rows: list[dict[str, Any]] | None = None,
    on_row: Any | None = None,
    through_action: int | None = None,
) -> list[dict[str, Any]]:
    if hasattr(alphabet, "idbs_for_pattern"):
        for i in range(alphabet.n_actions):
            alphabet.idbs_for_pattern(i)

    done_actions = {
        int(row["action"]) for row in (existing_rows or []) if "action" in row
    }
    rows: list[dict[str, Any]] = list(existing_rows or [])

    for action in range(alphabet.n_actions):
        if through_action is not None and action > through_action:
            break
        if action in done_actions:
            continue
        row = _row_from_step(
            env,
            alphabet,
            action=action,
            no_stim=no_stim,
            label_freq=label_freq,
        )
        rows.append(row)
        if on_row is not None:
            on_row(rows)
        if action % 20 == 0:
            freq = row.get("frequency_hz", action)
            print(
                f"  hz={freq:5.0f} action={action:3d}: Pβ={row['p_beta_raw']:.1f}",
                flush=True,
            )
    rows.sort(key=lambda r: int(r["action"]))
    return rows


def _summarize(rows: list[dict[str, Any]], *, no_stim: float, label: str) -> dict[str, Any]:
    beat = [r for r in rows if r["beat_no_stim"]]
    best = min(rows, key=lambda r: r["p_beta_raw"])
    return {
        "label": label,
        "n_frequencies": len(rows),
        "no_stim_p_beta": no_stim,
        "n_beat_no_stim": len(beat),
        "best_frequency_hz": float(best.get("frequency_hz", best["action"])),
        "best_action": int(best["action"]),
        "best_p_beta": float(best["p_beta_raw"]),
        "beaters": [
            {
                "frequency_hz": float(r.get("frequency_hz", r["action"])),
                "p_beta_raw": float(r["p_beta_raw"]),
            }
            for r in sorted(beat, key=lambda x: x["p_beta_raw"])
        ],
    }


def _load_checkpoint() -> dict[str, Any]:
    if not OUT_JSON.is_file():
        return {}
    return json.loads(OUT_JSON.read_text(encoding="utf-8"))


def _write_checkpoint(payload: dict[str, Any]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuous-only",
        action="store_true",
        help="Skip the discrete 41-action Kumaravelu grid comparison.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load partial JSON and skip actions already in continuous_1hz.landscape.",
    )
    parser.add_argument(
        "--through-action",
        type=int,
        default=None,
        metavar="N",
        help="Stop after completing action N (inclusive). For checkpoint backfill.",
    )
    add_max_threads_argument(parser)
    args = parser.parse_args()
    if args.max_threads is not None and args.max_threads < 1:
        parser.error("--max-threads must be >= 1")
    if (
        args.max_threads is not None
        and _BOOTSTRAP_MAX_THREADS is not None
        and args.max_threads != _BOOTSTRAP_MAX_THREADS
    ):
        parser.error("--max-threads was already applied at import; restart the process to change it")
    if args.max_threads is None:
        args.max_threads = _BOOTSTRAP_MAX_THREADS
    return args


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)

    checkpoint = _load_checkpoint() if args.resume else {}
    cont_existing = list(checkpoint.get("continuous_1hz", {}).get("landscape", []))

    print("=== TASK-177 exp 6: continuous frequency landscape 0–200 Hz ===", flush=True)
    banner = format_thread_limit_banner(args.max_threads)
    if banner:
        print(banner, flush=True)
    if args.resume:
        if cont_existing:
            last = max(int(r["action"]) for r in cont_existing)
            print(
                f"Resume: {len(cont_existing)} continuous actions on disk "
                f"(last completed action {last})",
                flush=True,
            )
        elif OUT_JSON.is_file():
            print("Resume: checkpoint JSON present but no continuous rows yet", flush=True)
        else:
            print("Resume: no checkpoint JSON — starting fresh", flush=True)

    no_stim = float(checkpoint.get("no_stim_p_beta", 0.0))
    if no_stim > 0.0 and args.resume:
        print(f"no-stim Pβ: {no_stim:.2f} (from checkpoint)", flush=True)
    else:
        ref_env = _make_env(
            plant_cfg=plant_cfg,
            alphabet=ContinuousFrequencyAlphabet(dt_ms=PAPER_DT_MS),
        )
        ref_env.reset(seed=PROBE_SEED)
        _obs, _r, _t, _tr, ref_info = ref_env.step(0)
        no_stim = float(ref_info["p_beta_raw"])
        ref_env.close()
        print(f"no-stim Pβ: {no_stim:.2f} (ref {NO_STIM_REF})", flush=True)

    result: dict[str, Any] = {
        "task": "TASK-177-exp6",
        "experiment": "continuous_scalar_frequency",
        "status": "in_progress",
        "continuous_only": args.continuous_only,
        "max_threads": args.max_threads,
        "plant_dt_ms": PAPER_DT_MS,
        "state_length": STATE_LENGTH,
        "reward_state_mode": "full_segment",
        "seed": PROBE_SEED,
        "no_stim_p_beta": no_stim,
        "continuous_1hz": {
            "landscape": cont_existing,
        },
        "elapsed_s": round(time.time() - t0, 2),
    }

    def _save_partial(rows: list[dict[str, Any]]) -> None:
        result["continuous_1hz"]["landscape"] = rows
        result["elapsed_s"] = round(time.time() - t0, 2)
        _write_checkpoint(result)

    print("\n--- Continuous 1 Hz grid (201 frequencies) ---", flush=True)
    cont_alpha = ContinuousFrequencyAlphabet(dt_ms=PAPER_DT_MS)
    cont_env = _make_env(plant_cfg=plant_cfg, alphabet=cont_alpha)
    try:
        cont_rows = _landscape(
            cont_env,
            cont_alpha,
            no_stim=no_stim,
            label_freq=True,
            existing_rows=cont_existing,
            on_row=_save_partial,
            through_action=args.through_action,
        )
    finally:
        cont_env.close()
    cont_summary = _summarize(cont_rows, no_stim=no_stim, label="continuous_1hz")
    result["continuous_1hz"]["summary"] = cont_summary

    if args.through_action is not None:
        result["status"] = "in_progress"
        result["elapsed_s"] = round(time.time() - t0, 2)
        _write_checkpoint(result)
        print(
            f"\nStopped at --through-action {args.through_action} "
            f"({len(cont_rows)} rows saved)",
            flush=True,
        )
        return 0

    disc_summary: dict[str, Any] | None = None
    disc_rows: list[dict[str, Any]] | None = None
    if not args.continuous_only:
        from envs.mehregan.patterns import PatternAlphabet

        print("\n--- Discrete Kumaravelu grid (41 actions, 5 Hz steps) ---", flush=True)
        disc_alpha = PatternAlphabet()
        disc_env = _make_env(plant_cfg=plant_cfg, alphabet=disc_alpha)
        try:
            disc_rows = _landscape(disc_env, disc_alpha, no_stim=no_stim)
        finally:
            disc_env.close()
        disc_summary = _summarize(disc_rows, no_stim=no_stim, label="discrete_5hz_grid")
        result["discrete_5hz_grid"] = {
            "summary": disc_summary,
            "landscape": disc_rows,
        }
        result["comparison"] = {
            "continuous_beats_no_stim": cont_summary["n_beat_no_stim"] > 0,
            "discrete_beats_no_stim": disc_summary["n_beat_no_stim"] > 0,
            "continuous_best_hz": cont_summary["best_frequency_hz"],
            "continuous_best_p_beta": cont_summary["best_p_beta"],
            "discrete_best_hz": disc_summary["best_frequency_hz"],
            "discrete_best_p_beta": disc_summary["best_p_beta"],
            "continuous_finds_lower_p_beta": cont_summary["best_p_beta"]
            < disc_summary["best_p_beta"],
        }

    result["status"] = "done"
    result["elapsed_s"] = round(time.time() - t0, 2)
    _write_checkpoint(result)

    print("\n=== Summary ===", flush=True)
    print(
        f"  Continuous: {cont_summary['n_beat_no_stim']}/201 beat no-stim; "
        f"best {cont_summary['best_frequency_hz']:.0f} Hz → Pβ={cont_summary['best_p_beta']:.1f}",
        flush=True,
    )
    if disc_summary is not None:
        print(
            f"  Discrete:   {disc_summary['n_beat_no_stim']}/41 beat no-stim; "
            f"best {disc_summary['best_frequency_hz']:.0f} Hz → Pβ={disc_summary['best_p_beta']:.1f}",
            flush=True,
        )
        print(json.dumps(result["comparison"], indent=2), flush=True)
    print(f"\nWrote {OUT_JSON} ({result['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
