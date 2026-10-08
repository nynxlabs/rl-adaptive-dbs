"""Probe: Nguyen Fig 7 with the "slip + carry" episode start (diagnostic only, not the panel run).

Two monkeypatches on the eval env, both guesses at unpublished environment details:

- Reset-order slip: the reset's first 100 ms reading runs at the previous episode's final DBS
  setting; the initial 40 Hz / 300 nA/cm^2 / 0.3 ms setting takes over from step 1. Episode 0
  uses the initial setting.
- Carried network (``--carry``): the plant keeps its neuron state across the episode reset; the
  per-episode seed still changes.

Every point is a plant reading. Best match so far (Oct 8 2026): the ``noterm-s4`` Fig 4 candidate
with both patches passes every Fig 7 gate. See docs/figures/nguyen/7.md
§ Reset-order slip with a carried network. Run from the repo root (the gates read the digitized
paper curve by relative path).

    .venv/bin/python scripts/probes/nguyen_fig7_slip_carry.py \\
        artifacts/figures/papers/nguyen/4/candidates/noterm-s4/checkpoint.pt \\
        --carry --out-dir artifacts/figures/papers/nguyen/7/candidates/slipcarry-noterm-s4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/digitization"))

import numpy as np  # noqa: E402

from controllers.snn import adapter as A  # noqa: E402
from controllers.snn.eval import evaluate  # noqa: E402
from envs.plant import python_backend as PB  # noqa: E402
from nguyen_gates import fig7_eval_gates  # noqa: E402

FIG3_PD_ON_MEDIAN = 295.2


def patch(*, slip: bool, carry: bool) -> list:
    """Install the patches; returns the list that collects each episode's first-step DBS setting."""
    cls = A.NguyenEnvAdapter
    orig_reset, orig_int = cls.reset, cls._integrate_current_dbs
    first: list = []

    def reset(self, *a, **kw):
        prev = getattr(self, "_dbs", None)
        self._stale_dbs = prev.copy() if (slip and prev is not None) else None
        return orig_reset(self, *a, **kw)

    def integrate(self):
        stale = getattr(self, "_stale_dbs", None)
        if stale is None:
            return orig_int(self)
        self._stale_dbs, init = None, self._dbs
        self._dbs = stale
        first.append((stale.frequency_hz, stale.amplitude, stale.pulse_width_ms))
        try:
            return orig_int(self)
        finally:
            self._dbs = init

    cls.reset, cls._integrate_current_dbs = reset, integrate

    if carry:
        orig_plant_reset = PB.PythonPlant.reset

        def plant_reset(self, seed=None):
            dyn = getattr(self, "_dyn", None)
            orig_plant_reset(self, seed)
            if dyn is not None:
                self._dyn = dyn
            return self

        PB.PythonPlant.reset = plant_reset
    return first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--no-slip", action="store_true", help="keep the reset order as shipped")
    ap.add_argument("--carry", action="store_true", help="keep neuron state across episode resets")
    ap.add_argument("--png-name", default="eval_50ep_slipcarry_v1.png")
    args = ap.parse_args()

    first = patch(slip=not args.no_slip, carry=args.carry)
    r = evaluate(args.checkpoint, episodes=args.episodes)
    trajs = r["alpha_beta_trajectories"]
    g = fig7_eval_gates(trajs, fig3_pd_on_median=FIG3_PD_ON_MEDIAN)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "eval.pkl", "wb") as f:
        pickle.dump({"eval": r, "first_dbs": first, "gates": g}, f)
    mean = np.asarray(trajs).mean(0)
    summary = {
        "checkpoint": str(args.checkpoint),
        "slip": not args.no_slip,
        "carry": args.carry,
        "episodes": args.episodes,
        "mean_steps_0_6": np.round(mean[:7]).astype(int).tolist(),
        "gates": g["gates"],
        "metrics": {k: v for k, v in g["metrics"].items() if np.isscalar(v)},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    spec = importlib.util.spec_from_file_location("p7", ROOT / "scripts/figures/papers/nguyen/7/plot.py")
    p7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p7)
    png = args.out_dir / args.png_name
    p7.plot_eval({"alpha_beta_trajectories": trajs}, png)

    fails = [k for k, v in g["gates"].items() if v is False]
    print("steps 0-6", summary["mean_steps_0_6"], "r", round(g["metrics"]["pearson_mean_trace"], 2))
    print("FAILS", fails)
    print(png)


if __name__ == "__main__":
    main()
