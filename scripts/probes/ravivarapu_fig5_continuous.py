#!/usr/bin/env python3
"""Fig 5 steps 0–5: continuous Kumaravelu trajectory vs independent 100 ms shots.

Each ``env.step()`` restarts the plant from the same ICs, so last-window Pβ
freezes after one stim. Paper Fig 5 is 10 sequential 2 ms steps. This probe
integrates one stretch (100 ms untreated + 500 ms stim) and measures Pβ on
non-overlapping 100 ms blocks and on a 100 ms sliding window (20 ms hop).

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_continuous.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs.plant.biomarkers import p_beta
from envs.plant.dbs import DbsSpec, create_dbs_current
from envs.plant.python_backend import PythonPlant

OUT = Path("artifacts/figures/papers/ravivarapu/5a/continuous_probe.json")
SCALE = 425.0
DT_MS = 0.01
UNTREATED_S = 0.1
STIM_S = 0.5
HOP_S = 0.02
WIN_S = 0.1
PAPER_5A_SEA = (0.4578, 0.4427, 0.4295, 0.3960, 0.3627, 0.3381)
PAPER_5A_B = (0.4604, 0.4531, 0.4420, 0.4305, 0.4177, 0.4037)


def _idbs(*, hz: float, burst_ms: float) -> np.ndarray:
    tmax_ms = (UNTREATED_S + STIM_S) * 1000.0
    full = create_dbs_current(hz, tmax_ms=tmax_ms, dt_ms=DT_MS)
    n_unt = int(round(UNTREATED_S * 1000.0 / DT_MS))
    full[:n_unt] = 0.0
    if burst_ms >= 100.0:
        return full
    # Repeat Fig 4a short burst inside each 100 ms stim block.
    block = int(round(100.0 / DT_MS))
    keep = int(round(burst_ms / DT_MS))
    i = n_unt
    while i < full.size:
        full[i + keep : i + block] = 0.0
        i += block
    return full


def _window_norm(gpi_spikes: list[np.ndarray], t0: float, t1: float) -> float:
    dur = t1 - t0
    clipped = []
    for sp in gpi_spikes:
        sp = np.asarray(sp, dtype=float)
        w = sp[(sp >= t0) & (sp < t1)] - t0
        clipped.append(w)
    raw = p_beta(clipped, dt_ms=DT_MS, segment_duration_s=dur)
    return float(raw) / SCALE


def _mae(got: list[float], ref: tuple[float, ...]) -> float:
    n = min(len(got), len(ref))
    return sum(abs(got[i] - ref[i]) for i in range(n)) / n


def _run(*, hz: float, burst_ms: float, seed: int = 0) -> dict:
    plant = PythonPlant()
    plant.reset(seed=seed)
    idbs = _idbs(hz=hz, burst_ms=burst_ms)
    result = plant.integrate(
        UNTREATED_S + STIM_S,
        DbsSpec(pick_dbs_freq=2, idbs=idbs),
        record_spikes=True,
    )
    spikes = result.gpi_spikes
    blocks = [
        _window_norm(spikes, i * WIN_S, (i + 1) * WIN_S) for i in range(6)
    ]
    sliding = [
        _window_norm(spikes, k * HOP_S, k * HOP_S + WIN_S) for k in range(6)
    ]
    return {
        "carrier_hz": hz,
        "dbs_burst_ms": burst_ms,
        "blocks_100ms": blocks,
        "sliding_hop20ms": sliding,
        "block_drop_0_5": blocks[0] - blocks[-1],
        "slide_drop_0_5": sliding[0] - sliding[-1],
        "mae_block_vs_paper_sea": _mae(blocks, PAPER_5A_SEA),
        "mae_slide_vs_paper_sea": _mae(sliding, PAPER_5A_SEA),
        "mae_block_vs_paper_b": _mae(blocks, PAPER_5A_B),
        "mae_slide_vs_paper_b": _mae(sliding, PAPER_5A_B),
    }


def main() -> None:
    rows = []
    for hz in (50.0, 30.0):
        for burst in (62.0, 100.0):
            row = _run(hz=hz, burst_ms=burst)
            rows.append(row)
            b = [round(x, 4) for x in row["blocks_100ms"]]
            s = [round(x, 4) for x in row["sliding_hop20ms"]]
            print(
                f"{hz:.0f} Hz burst={burst:.0f} "
                f"blocks={b} drop={row['block_drop_0_5']:.4f} "
                f"MAE_SEA={row['mae_block_vs_paper_sea']:.4f}"
            )
            print(
                f"         slide={s} drop={row['slide_drop_0_5']:.4f} "
                f"MAE_SEA={row['mae_slide_vs_paper_sea']:.4f}"
            )
    payload = {
        "paper_5a_sea": PAPER_5A_SEA,
        "paper_5a_baseline": PAPER_5A_B,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
