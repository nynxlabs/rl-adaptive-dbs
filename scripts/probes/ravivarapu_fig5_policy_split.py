#!/usr/bin/env python3
"""Fig 5 policy-split diagnostic (no plant).

Greedy Fig 4a actors are always-on at inference, so SEA vs Baseline traces
match. This probe reports logit margins and Gumbel-max stim rates. Hard
Gumbel-max is temperature-invariant; P(stim) = softmax(logits). That is the
cheapest honest duty-cycle split without retraining.

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_policy_split.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
from controllers.sea_dbs.networks import gumbel_softmax_sample

CKPT_DIR = Path("artifacts/figures/papers/ravivarapu/4")
OUT = Path("artifacts/figures/papers/ravivarapu/5a/policy_split_probe.json")
P_GRID = np.round(np.linspace(0.20, 0.50, 13), 3)
N_GS = 200


def _ckpt(variant: str) -> Path:
    path = CKPT_DIR / f"{variant}_train0.pt"
    if not path.is_file():
        raise SystemExit(f"missing checkpoint: {path}")
    return path


def _load(variant: str):
    payload = load_checkpoint(_ckpt(variant), device="cpu")
    actor, cfg = load_actor_from_payload(payload, device="cpu")
    actor.eval()
    return actor, cfg


def _logits(actor, p_beta: float) -> np.ndarray:
    state = torch.tensor([[p_beta]], dtype=torch.float32)
    with torch.no_grad():
        return actor(state).squeeze(0).cpu().numpy().astype(np.float32)


def _gs_stim_rate(actor, p_beta: float, tau: float, n: int, seed: int) -> float:
    torch.manual_seed(seed)
    state = torch.tensor([[p_beta]], dtype=torch.float32)
    stim = 0
    with torch.no_grad():
        logits = actor(state)
        for _ in range(n):
            _, index = gumbel_softmax_sample(logits, tau=tau, hard=True)
            stim += int(index.item())
    return stim / n


def main() -> None:
    rows = []
    for variant in ("baseline", "paper"):
        actor, cfg = _load(variant)
        tau_min = float(cfg.gs_tau_min)
        late_floor = float(getattr(cfg, "gs_late_tau_floor", tau_min) or tau_min)
        taus = sorted({round(tau_min, 4), round(late_floor, 4), 0.42, 0.87, 0.94, 1.0})
        logit_rows = []
        for p in P_GRID:
            z = _logits(actor, float(p))
            margin = float(z[1] - z[0])
            greedy = int(np.argmax(z))
            gs = {f"tau_{tau:g}": _gs_stim_rate(actor, float(p), tau, N_GS, seed=0) for tau in taus}
            logit_rows.append(
                {
                    "p_beta": float(p),
                    "logit0": float(z[0]),
                    "logit1": float(z[1]),
                    "margin_stim_minus_nostim": margin,
                    "greedy": greedy,
                    "softmax_tau1": F.softmax(torch.tensor(z), dim=-1)[1].item(),
                    **gs,
                }
            )
        rows.append(
            {
                "variant": variant,
                "gs_tau_min": tau_min,
                "gs_late_tau_floor": late_floor,
                "force_gumbel_softmax": bool(cfg.force_gumbel_softmax),
                "logits": logit_rows,
            }
        )

    payload = {"n_gs": N_GS, "p_grid": P_GRID.tolist(), "variants": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("summary (p=0.45; hard GS P(stim)=softmax, tau-invariant):")
    for block in rows:
        at = next(r for r in block["logits"] if abs(r["p_beta"] - 0.45) < 1e-9)
        print(
            block["variant"],
            f"greedy={at['greedy']}",
            f"margin={at['margin_stim_minus_nostim']:.3f}",
            f"softmax={at['softmax_tau1']:.3f}",
            f"gs_rate={at['tau_1']:.3f}",
        )


if __name__ == "__main__":
    main()
