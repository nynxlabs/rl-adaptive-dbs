#!/usr/bin/env python3
"""Cheap Fig 4a gate probe — baseline vs paper episode_psd slopes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_PLOT = ROOT / "scripts" / "figures" / "papers" / "ravivarapu" / "4a" / "plot.py"
_spec = importlib.util.spec_from_file_location("ravivarapu_4a_plot", _PLOT)
assert _spec and _spec.loader
_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/probes/ravivarapu_fig4a_gate_probe.json"))
    args = parser.parse_args()

    series: dict[str, object] = {"seed": args.seed, "smoke": False, "variants": {}}
    for variant in _plot.VARIANTS:
        print(f"training variant={variant} seed={args.seed} episodes={args.episodes}", flush=True)
        series["variants"][variant] = _plot.train_variant(
            variant,
            seed=args.seed,
            smoke=False,
            num_episodes=args.episodes,
        )

    gates = _plot.evaluate_gates(series)
    for variant in ("baseline", "paper"):
        psd = np.asarray(series["variants"][variant]["episode_psd"], dtype=float)  # type: ignore[index]
        slope = float(np.polyfit(np.arange(len(psd)), psd, 1)[0])
        print(
            f"{variant}: slope={slope:.6f} tail={psd[len(psd) // 2 :].mean():.4f}",
            flush=True,
        )
    print(json.dumps(gates, indent=2), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"episodes": args.episodes, "seed": args.seed, "gates": gates, "series": series}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
