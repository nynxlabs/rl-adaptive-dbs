#!/usr/bin/env python3
"""Fig 6: independent per-step integrates vs one continuous stitched trajectory.

Current ``SEA_DBSEnvAdapter`` calls ``integrate()`` once per RL step, but each
call cold-starts from the same cached IC draw — state does not accumulate. That
freezes always-on stim on the last-window floor. This probe stitches one long
50 Hz timeline (100 ms untreated onset + variable skip/stim segments from
Gumbel actions) and samples Pβ at each step boundary on the shared spike train.

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig6_continuous.py
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from controllers.sea_dbs.config import (
    FIG5A_GUMBEL_SEED_OFFSET,
    FIG5A_INFERENCE_BURST_MS,
    FIG5A_INFERENCE_WINDOW_S,
    INFERENCE_CARRIER_50HZ,
    OBSERVATION_SCALE,
    SEADBSConfig,
)
from controllers.sea_dbs.eval import evaluate
from envs.plant.biomarkers import p_beta
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.python_backend import PythonPlant

OUT = Path("artifacts/figures/papers/ravivarapu/6/continuous_probe.json")
DIG = Path("artifacts/figures/papers/ravivarapu/paper_digitization/curves_fig6.json")
SERIES = Path("artifacts/figures/papers/ravivarapu/6/series.json")
CKPT_BASE = Path("artifacts/figures/papers/ravivarapu/4/baseline_train0.pt")
CKPT_SEA = Path("artifacts/figures/papers/ravivarapu/4/paper_train0.pt")

DT_MS = 0.01
UNTREATED_S = 0.10
STIM_S = FIG5A_INFERENCE_WINDOW_S
BURST_MS = FIG5A_INFERENCE_BURST_MS
HZ = INFERENCE_CARRIER_50HZ
N_OBS = 8
GUMBEL_OFFSET = FIG5A_GUMBEL_SEED_OFFSET
SCALE = OBSERVATION_SCALE


def _n_steps(duration_s: float) -> int:
    return int(round(duration_s * 1000.0 / DT_MS)) + 1


def _segment_idbs(*, action: int, duration_s: float) -> np.ndarray:
    n = _n_steps(duration_s)
    if int(action) == 0:
        return np.zeros(n, dtype=np.float64)
    full = create_dbs_current(HZ, tmax_ms=duration_s * 1000.0, dt_ms=DT_MS)
    if BURST_MS < duration_s * 1000.0:
        keep = int(round(BURST_MS / DT_MS))
        full[keep:] = 0.0
    return full


def _duration_s_for_action(action: int) -> float:
    return UNTREATED_S if int(action) == 0 else STIM_S


def _stitch_idbs(actions: list[int]) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """Return stitched idbs and segment metadata (t0_s, t1_s, action)."""
    parts: list[np.ndarray] = []
    segments: list[tuple[float, float, int]] = []
    t = 0.0
    dur = UNTREATED_S
    parts.append(_segment_idbs(action=0, duration_s=dur))
    segments.append((t, t + dur, 0))
    t += dur
    for action in actions:
        dur = _duration_s_for_action(action)
        parts.append(_segment_idbs(action=action, duration_s=dur))
        segments.append((t, t + dur, int(action)))
        t += dur
    idbs = parts[0]
    for part in parts[1:]:
        # Segment grids share endpoints; drop duplicate join sample.
        idbs = np.concatenate([idbs, part[1:]])
    return idbs, segments


def _window_p_beta(
    gpi_spikes: list[np.ndarray],
    *,
    t0: float,
    t1: float,
) -> float:
    clipped: list[np.ndarray] = []
    for sp in gpi_spikes:
        sp = np.asarray(sp, dtype=float)
        w = sp[(sp >= t0) & (sp < t1)] - t0
        clipped.append(w)
    dur = t1 - t0
    return float(p_beta(clipped, dt_ms=DT_MS, segment_duration_s=dur))


def _trailing_p_beta(
    gpi_spikes: list[np.ndarray],
    *,
    t_end: float,
    window_s: float,
) -> float:
    t0 = max(0.0, t_end - window_s)
    return _window_p_beta(gpi_spikes, t0=t0, t1=t_end)


def _n_obs_mean(values: list[float], n_obs: int) -> list[float]:
    window: deque[float] = deque(maxlen=n_obs)
    out: list[float] = []
    for v in values:
        window.append(v)
        out.append(float(np.mean(window)))
    return out


def _unique_levels(y: list[float], *, nd: int = 1) -> int:
    return len({round(float(v), nd) for v in y})


def _turning_points(y: list[float], *, min_dy: float = 0.5) -> int:
    arr = np.asarray(y, dtype=float)
    dy = np.diff(arr)
    sig = np.sign(np.where(np.abs(dy) < min_dy, 0.0, dy))
    nz = sig[sig != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def _paper_integer_steps() -> dict[str, list[float]]:
    dig = json.loads(DIG.read_text(encoding="utf-8"))
    xs = np.arange(0, 11, dtype=float)
    out: dict[str, list[float]] = {}
    for name, s in dig["series"].items():
        x = np.asarray(s["xy"]["x"], dtype=float)
        y = np.asarray(s["xy"]["y"], dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        _, uidx = np.unique(x, return_index=True)
        x, y = x[uidx], y[uidx]
        out[name] = [float(v) for v in np.interp(xs, x, y)]
    return out


def _mae(got: list[float], ref: list[float]) -> float:
    n = min(len(got), len(ref))
    return float(np.mean(np.abs(np.asarray(got[:n]) - np.asarray(ref[:n]))))


def _actions_for_checkpoint(path: Path, *, variant: str, seed: int = 0, steps: int = 10) -> list[int]:
    payload = evaluate(
        path,
        config=SEADBSConfig(variant=variant, seed=seed),
        max_steps=steps,
        carrier_hz=HZ,
        action_mode="gumbel",
        dbs_burst_ms=BURST_MS,
        biomarker_window_s=STIM_S,
        n_obs=N_OBS,
        gumbel_seed_offset=GUMBEL_OFFSET,
        untreated_window_s=UNTREATED_S,
    )
    return [int(a) for a in payload["action_trajectories"][0]]


def _continuous_trace(
    actions: list[int],
    *,
    seed: int,
    sample: str,
) -> dict[str, Any]:
    idbs, segments = _stitch_idbs(actions)
    plant = PythonPlant()
    plant.reset(seed=seed)
    result = plant.integrate(
        segments[-1][1],
        DbsSpec(pick_dbs_freq=2, idbs=idbs),
        record_spikes=True,
    )
    if not result.gpi_spikes:
        msg = "continuous integrate missing gpi_spikes"
        raise RuntimeError(msg)

    raw_segment: list[float] = []
    for t0, t1, action in segments:
        if sample == "segment_last":
            raw_segment.append(_window_p_beta(result.gpi_spikes, t0=t0, t1=t1))
        elif sample == "trailing_stim":
            win = UNTREATED_S if action == 0 else STIM_S
            raw_segment.append(_trailing_p_beta(result.gpi_spikes, t_end=t1, window_s=win))
        elif sample == "trailing_150":
            raw_segment.append(_trailing_p_beta(result.gpi_spikes, t_end=t1, window_s=STIM_S))
        else:
            raise ValueError(sample)

    norm_segment = [v / SCALE for v in raw_segment]
    norm_nobs = _n_obs_mean(norm_segment, N_OBS)
    display = [v * 1000.0 for v in norm_nobs]
    return {
        "sample": sample,
        "actions": actions,
        "raw_segment": [round(v, 4) for v in raw_segment],
        "norm_nobs": [round(v, 6) for v in norm_nobs],
        "display_x1000": [round(v, 1) for v in display],
        "unique_levels": _unique_levels(display),
        "turning_points": _turning_points(display),
        "last3": display[-3:],
        "last_half_mean": float(np.mean(display[6:])),
    }


def main() -> None:
    if SERIES.is_file():
        cached = json.loads(SERIES.read_text(encoding="utf-8"))
        actions = {k: [int(x) for x in v] for k, v in cached["actions"].items()}
        independent = {
            label: [round(float(x) * 1000.0, 1) for x in cached["traces"][label]]
            for label in cached["traces"]
        }
    else:
        actions = {
            "Baseline": _actions_for_checkpoint(CKPT_BASE, variant="baseline"),
            "SEA-DBS": _actions_for_checkpoint(CKPT_SEA, variant="paper"),
        }
        independent = {}

    paper = _paper_integer_steps()
    rows: list[dict[str, Any]] = []
    for label in ("Baseline", "SEA-DBS", "SEA-DBS + PTQ(fp16)"):
        if label not in actions:
            continue
        act = actions[label]
        for sample in ("segment_last", "trailing_stim", "trailing_150"):
            row = _continuous_trace(act, seed=0, sample=sample)
            row["label"] = label
            paper_key = label
            row["mae_vs_paper"] = round(_mae(row["display_x1000"], paper[paper_key]), 2)
            row["paper_last3"] = paper[paper_key][-3:]
            if label in independent:
                row["independent_last3"] = independent[label][-3:]
                row["mae_vs_independent"] = round(
                    _mae(row["display_x1000"], independent[label]), 2
                )
            rows.append(row)

    payload = {
        "n_obs": N_OBS,
        "untreated_s": UNTREATED_S,
        "stim_s": STIM_S,
        "burst_ms": BURST_MS,
        "carrier_hz": HZ,
        "actions": actions,
        "paper_display": {k: [round(v, 1) for v in vals] for k, vals in paper.items()},
        "independent_v14_display": independent,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {OUT}\n")
    print("=== paper (digitized, steps 0-10) ===")
    for name in ("Baseline", "SEA-DBS", "SEA-DBS + PTQ(fp16)"):
        y = paper[name]
        print(
            f"{name:24s} unique={_unique_levels(y)} turns={_turning_points(y)} "
            f"last3={[round(v,1) for v in y[-3:]]}"
        )
    if independent:
        print("\n=== independent v14 (current eval) ===")
        for name, y in independent.items():
            print(
                f"{name:24s} unique={_unique_levels(y)} turns={_turning_points(y)} "
                f"last3={y[-3:]}"
            )
    print("\n=== continuous probes ===")
    for row in rows:
        print(
            f"{row['label']:24s} {row['sample']:14s} "
            f"unique={row['unique_levels']} turns={row['turning_points']} "
            f"MAE={row['mae_vs_paper']:5.1f} last3={row['last3']}"
        )


if __name__ == "__main__":
    main()
