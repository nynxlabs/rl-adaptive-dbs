"""Probe: Nguyen Fig 7 onset on the plant alone (diagnostic only, not the panel run).

Holds one DBS setting open-loop from a fresh network for ``--steps`` 100 ms segments, on the
same 50 per-episode seeds the Fig 7 eval uses, with the plant settings taken from a Fig 4
checkpoint (carried plant, STN recruitment spread, observed populations). No policy acts, so
step 0 here is exactly the eval's step 0 (the reset reading), and later steps show what the
network does by itself. Conditions:

- ``pd_init``: Parkinsonian network at the paper's initial 40 Hz / 300 nA/cm^2 / 0.3 ms setting
- ``pd_off``: Parkinsonian network, stimulation off (amplitude 0)
- ``healthy_off``: healthy network (``pd = 0``), stimulation off

For each segment it records GPi alpha-beta (the eval's measure), the GPi spike count, and for
steps 0-2 a 10 ms GPi spike histogram. See docs/figures/nguyen/7.md § Onset on the
distance-based plant. Run from the repo root:

    RL_DBS_MAX_THREADS=1 .venv/bin/python scripts/probes/nguyen_fig7_onset_map.py \\
        artifacts/figures/papers/nguyen/4/checkpoint.pt \\
        --out artifacts/figures/papers/nguyen/7/candidates/onset-map-rho3/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from controllers.snn import adapter as A  # noqa: E402
from controllers.snn.config import SNNConfig  # noqa: E402
from controllers.snn.dbs_params import DBSParameterState  # noqa: E402
from controllers.snn.reward import alpha_beta_power  # noqa: E402
from controllers.snn.trainer import load_checkpoint  # noqa: E402

CONDITIONS = {
    "pd_init": {"pd": 1, "amplitude": None},
    "pd_off": {"pd": 1, "amplitude": 0.0},
    "healthy_off": {"pd": 0, "amplitude": 0.0},
}


def run_condition(cfg: SNNConfig, name: str, *, episodes: int, steps: int) -> dict:
    spec = CONDITIONS[name]
    env = A.NguyenEnvAdapter(config=cfg)
    env._plant.config = replace(env._plant.config, pd=spec["pd"])
    orig_from_config = DBSParameterState.from_config

    def from_config(_cls, config=None):
        state = orig_from_config(config)
        if spec["amplitude"] is not None:
            state.amplitude = float(spec["amplitude"])
        return state

    A.DBSParameterState.from_config = classmethod(from_config)
    ab = np.zeros((episodes, steps))
    counts = np.zeros((episodes, steps))
    hist = np.zeros((episodes, 3, 10))
    try:
        for ep in range(episodes):
            env.reset(seed=cfg.seed + 10_000 + ep)
            # reset() already integrated step 0; re-run it here to read its spikes.
            env._plant.reset(seed=cfg.seed + 10_000 + ep)
            env._pulses.reset()
            for step in range(steps):
                result = env._integrate_current_dbs()
                gpi = env._gpi_spike_trains(result)
                ab[ep, step] = alpha_beta_power(gpi, duration_s=cfg.step_duration_s, dt_ms=result.dt_ms)
                times = np.concatenate([np.asarray(t, dtype=float) for t in gpi]) if gpi else np.array([])
                counts[ep, step] = times.size
                if step < 3 and times.size:
                    local = times - np.floor(times / cfg.step_duration_s + 1e-9) * cfg.step_duration_s
                    hist[ep, step] = np.histogram(local, bins=10, range=(0.0, cfg.step_duration_s))[0]
    finally:
        A.DBSParameterState.from_config = classmethod(lambda _cls, config=None: orig_from_config(config))
        env.close()
    return {
        "alpha_beta_mean": np.round(ab.mean(0), 1).tolist(),
        "alpha_beta_ci95": np.round(1.96 * ab.std(0, ddof=1) / np.sqrt(episodes), 1).tolist(),
        "gpi_spikes_mean": np.round(counts.mean(0), 2).tolist(),
        "gpi_hist_10ms_steps_0_2": np.round(hist.mean(0), 2).tolist(),
        "alpha_beta_per_spike": float(np.sum(ab) / max(np.sum(counts), 1.0)),
        "corr_alpha_beta_spikes": float(np.corrcoef(ab.ravel(), counts.ravel())[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()

    saved = load_checkpoint(args.checkpoint, map_location="cpu")["config"]
    cfg = saved.with_variant_defaults()
    out = {"checkpoint": str(args.checkpoint), "stn_dbs_spread": cfg.stn_dbs_spread, "episodes": args.episodes}
    for name in args.conditions:
        out[name] = run_condition(cfg, name, episodes=args.episodes, steps=args.steps)
        print(name, "alpha-beta", out[name]["alpha_beta_mean"], flush=True)
        print(name, "GPi spikes", out[name]["gpi_spikes_mean"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
