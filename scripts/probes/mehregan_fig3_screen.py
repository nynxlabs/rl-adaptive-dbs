"""Screen paper-silent knobs on the Mehregan Fig 3 actor (Figs 4a / 5a / 5b).

One run: train the Fig 3 actor with ``paper_protocol.train_paper`` at one mean
rate and seed, then evaluate on the eval seed, closed loop, both the actor as
initialized and the trained actor, next to no stimulation and the periodic
train. Separates what training learned (init vs trained deterministic policy)
from what the exploration schedule does to the training trace. Per episode it
also logs how many pulses the noise-free actor puts on their regular slot and
the spread of the logits. At 45 Hz it scores the training trace with the Fig 4a
gates. Diagnostic only.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_fig3_screen.py \
      --mean-hz 45 --seed 0 --set gamma=0.5 --out artifacts/probes/mehregan_fig3/screen_g05.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_adaptive_dbs.panel import load_script_module

_ROOT = Path(__file__).resolve().parents[1]
_PP = load_script_module(
    "mehregan_paper_protocol", _ROOT / "figures/papers/mehregan/paper_protocol.py"
)
sys.path.insert(0, str(_ROOT / "digitization"))
from paper_gates import fig4a_gates  # noqa: E402

REF_LEVELS = (0.40, 0.50)


def _actor_diag(actor: torch.nn.Module, space: Any) -> dict[str, Any]:
    reg = space.regular_slots()
    rows = np.arange(space.n_pulses)
    out: dict[str, Any] = {}
    with torch.no_grad():
        for level in REF_LEVELS:
            logits = actor(torch.full((1, _PP.STATE_LENGTH), level)).squeeze(0).numpy()
            grid = logits.reshape(space.n_pulses, space.n_slots)
            out[f"{level:.2f}"] = {
                "frac_regular_argmax": float(np.mean(grid.argmax(axis=1) == reg)),
                "regular_logit_mean": float(grid[rows, reg].mean()),
                "logit_std": float(grid.std()),
                "row_max_mean": float(grid.max(axis=1).mean()),
            }
    return out


def _cond_summary(cond: dict[str, Any]) -> dict[str, Any]:
    pre, late = _PP.pre_mean(cond), _PP.post_mean(cond, lo=4.0)
    out = {"pre": pre, "late": late, "late_over_pre": late / pre}
    if cond.get("actions"):
        out["distinct_slots"] = [len(set(r)) for r in cond["actions"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hz", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--variant", default="paper", choices=("paper", "qat"))
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
    space = _PP.make_alphabet(args.mean_hz, knobs)
    init: dict[str, torch.nn.Module] = {}
    diag: list[dict[str, Any]] = []

    def on_start(agent: Any) -> None:
        init["actor"] = copy.deepcopy(agent.actor)
        diag.append({"episode": -1, **_actor_diag(agent.actor, space)})

    def on_episode(agent: Any, episode: int) -> None:
        d = {"episode": episode, **_actor_diag(agent.actor, space)}
        diag.append(d)
        print(f"  actor after ep{episode}: {d['0.50']}", flush=True)

    t0 = time.time()
    record = _PP.train_paper(
        mean_hz=args.mean_hz, seed=args.seed, knobs=knobs, variant=args.variant,
        checkpoint_path=args.checkpoint, trainer_hook=on_start, episode_hook=on_episode,
    )
    train_s = time.time() - t0
    load = _PP.load_fp32_actor if args.variant == "paper" else _PP.load_qat_actor
    trained = load(args.checkpoint)
    evals: dict[str, Any] = {}
    run = lambda **kw: _PP.run_condition(seed=args.eval_seed, alphabet=space, knobs=knobs, **kw)  # noqa: E731
    evals["no_stim"] = _cond_summary(run())
    evals["periodic"] = _cond_summary(run(fixed_spec=_PP.regular_spec(space)))
    evals["init_actor"] = _cond_summary(run(policy=_PP.module_policy(init["actor"], space)))
    evals["trained_actor"] = _cond_summary(run(policy=_PP.module_policy(trained, space)))
    steps = _PP.STEPS_PER_EPISODE
    per_ep = lambda xs: [float(np.mean(xs[i:i + steps])) for i in range(0, len(xs), steps)]  # noqa: E731
    out: dict[str, Any] = {
        "mean_hz": args.mean_hz,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "variant": args.variant,
        "knobs": knobs.as_dict(),
        "train_s": round(train_s, 1),
        "beta_per_episode": per_ep(record["beta_trace"]),
        "isi_std_per_episode": per_ep(record["isi_std_ms"]),
        "episode_rewards": record["episode_rewards"],
        "beta_trace": record["beta_trace"],
        "actor_diag": diag,
        "evals": evals,
    }
    if args.mean_hz == 45.0:
        g = fig4a_gates(record["beta_trace"], n_expected=len(record["beta_trace"]))
        out["fig4a_gates"] = g["gates"]
        out["fig4a_metrics"] = g["metrics"]
        out["fig4a_pass"] = bool(g["pass"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"beta/episode {[round(v, 3) for v in out['beta_per_episode']]}", flush=True)
    print(f"evals {json.dumps({k: round(v['late_over_pre'], 3) for k, v in evals.items()})}", flush=True)
    if "fig4a_gates" in out:
        print(f"fig4a {out['fig4a_gates']} pass={out['fig4a_pass']}", flush=True)
    print(f"wrote {args.out} ({train_s:.0f}s train)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
