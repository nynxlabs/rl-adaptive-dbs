"""Open-loop pattern landscape under the Mehregan paper eval protocol.

Each pattern of the alphabet is played for all five 2 s eval steps on a fixed
seed (plant state carried); reports the post-onset trailing $P_\\beta$ mean
(display t >= 4 s) next to no-stim. Diagnostic only.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_paper_landscape.py \
      --mean-hz 45 --alphabet burst --out artifacts/probes/mehregan_landscape_45_burst.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rl_adaptive_dbs.panel import load_script_module

_PP = load_script_module(
    "mehregan_paper_protocol",
    Path(__file__).resolve().parents[1] / "figures/papers/mehregan/paper_protocol.py",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hz", type=float, required=True)
    ap.add_argument("--alphabet", default="burst")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    knobs = _PP.apply_overrides(_PP.PaperKnobs(), [f"alphabet={args.alphabet}"])
    alphabet = _PP.make_alphabet(args.mean_hz, knobs)
    seeds = [int(s) for s in args.seeds.split(",")]
    out: dict = {"mean_hz": args.mean_hz, "alphabet": args.alphabet, "seeds": seeds, "rows": {}}
    t0 = time.time()
    for seed in seeds:
        rows = {}
        ns = _PP.run_condition(seed=seed, alphabet=alphabet)
        rows["no_stim"] = _PP.post_mean(ns)
        for a in range(alphabet.n_actions):
            c = _PP.run_condition(seed=seed, alphabet=alphabet, fixed_spec=alphabet.to_dbs_spec(a))
            rows[str(a)] = _PP.post_mean(c)
            print(f"seed {seed} pattern {a}: {rows[str(a)]:.1f} (no_stim {rows['no_stim']:.1f})", flush=True)
        out["rows"][str(seed)] = rows
    out["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
