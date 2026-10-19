#!/usr/bin/env python3
"""Can post-training quantization flip a SEA-DBS actor's action? (Fig 6 candidate.)

The paper's Fig 6 PTQ is fp16 (§V). This probe compares, over a grid of Eq. (5)
states, the fp32 actor with (a) the fp16 cast the panel uses and (b) a static,
calibrated int8 PTQ (per-tensor symmetric weights; activations quantized with
scales calibrated on the same state grid — the usual "static PTQ" recipe), a
candidate reading of what the authors may have run. An action can only flip where
the quantization error in the stim − no-stim logit margin exceeds the margin.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_ptq_static.py CKPT [CKPT ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
from controllers.sea_dbs.quantization import FP16ActorWrapper, apply_fp16_ptq


def _q(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def int8_static_forward(net: nn.Sequential, x: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
    """Fake-quant int8: weights per tensor, activations with scales from ``calib``."""
    h, hc = x, calib
    for layer in net:
        if isinstance(layer, nn.Linear):
            h = _q(h, float(hc.abs().max()) / 127 or 1.0)
            w = _q(layer.weight, float(layer.weight.abs().max()) / 127)
            h = h @ w.T + layer.bias
            hc = layer(hc)
        else:
            h, hc = layer(h), layer(hc)
    return h


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--lo", type=float, default=0.25)
    parser.add_argument("--hi", type=float, default=0.60)
    args = parser.parse_args()
    out = {}
    for ckpt in args.checkpoints:
        actor, cfg = load_actor_from_payload(load_checkpoint(ckpt))
        if cfg.obs_dim != 1:
            print(f"{ckpt}: obs_dim {cfg.obs_dim} — grid probe covers mean-only states; skipped")
            continue
        x = torch.linspace(args.lo, args.hi, 701).reshape(-1, 1)
        with torch.no_grad():
            l32 = actor(x)
            l16 = FP16ActorWrapper(apply_fp16_ptq(actor))(x).float()
            l8 = int8_static_forward(actor.net, x, x)
        m32 = (l32[:, 1] - l32[:, 0]).numpy()
        row = {
            "min_abs_margin": float(np.abs(m32).min()),
            "margin_range": [float(m32.min()), float(m32.max())],
            "stim_frac_fp32": float((m32 > 0).mean()),
        }
        for name, lq in (("fp16", l16), ("int8_static", l8)):
            mq = (lq[:, 1] - lq[:, 0]).numpy()
            row[f"{name}_max_margin_err"] = float(np.abs(mq - m32).max())
            row[f"{name}_flips"] = int(((mq > 0) != (m32 > 0)).sum())
        out[ckpt.as_posix()] = row
        print(ckpt, json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
