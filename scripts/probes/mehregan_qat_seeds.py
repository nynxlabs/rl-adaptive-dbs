"""Where does 10-episode QAT land across training seeds? (Mehregan Fig 6a/6b)

QAT trains with the paper recipe; its logits saturate at the fake-quant
clamp, so the deployed argmax often ties and resolves to the lowest tied
index. This probe trains QAT per seed and records the eval action, the size
of the tie, and the closed-loop late/pre level against fp32-like suppression.
Diagnostic only.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_qat_seeds.py \
      --mean-hz 45 --seeds 1-3 --set logit_noise_std=1.0 --out artifacts/probes/qat45_1-3.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from rl_adaptive_dbs.panel import load_script_module

_PP = load_script_module(
    "mehregan_paper_protocol",
    Path(__file__).resolve().parents[1] / "figures/papers/mehregan/paper_protocol.py",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hz", type=float, required=True)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
    lo, _, hi = args.seeds.partition("-")
    alphabet = _PP.make_alphabet(args.mean_hz, knobs)
    rows = {}
    for seed in range(int(lo), int(hi or lo) + 1):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "qat.pt"
            _PP.train_paper(
                mean_hz=args.mean_hz, seed=seed, knobs=knobs, variant="qat", checkpoint_path=ckpt
            )
            qat = _PP.load_qat_actor(ckpt)
        cond = _PP.run_condition(
            seed=args.eval_seed, alphabet=alphabet, policy=_PP.module_policy(qat)
        )
        logits = np.asarray(cond["logits"][0])
        row = {
            "actions": cond["actions"],
            "n_tied_at_max": int(np.sum(np.isclose(logits, logits.max()))),
            "pre": _PP.pre_mean(cond),
            "late": _PP.post_mean(cond, lo=4.0),
        }
        row["late_over_pre"] = row["late"] / row["pre"]
        rows[str(seed)] = row
        print(f"seed {seed}: {row}", flush=True)
    out = {"mean_hz": args.mean_hz, "knobs": knobs.as_dict(), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
