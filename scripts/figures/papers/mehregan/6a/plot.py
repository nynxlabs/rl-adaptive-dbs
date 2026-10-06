#!/usr/bin/env python3
"""Mehregan et al. Figure 6a — PTQ / QAT @ 45 Hz.

Thin entry point over ``mehregan/quant_panel.py`` (shared with Fig 6b). The fp32
series is the Fig 4a 45 Hz checkpoint; PTQ fp16 / int8 quantize it; QAT is a
separate 10-episode quantization-aware run with the same recipe.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/6a/plot.py --train-qat
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/6a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/6a/plot.py --plot-only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from rl_adaptive_dbs.panel import load_script_module  # noqa: E402

_QP = load_script_module(
    "mehregan_quant_panel", Path(__file__).resolve().parents[1] / "quant_panel.py"
)
CFG = _QP.FIG6A

if __name__ == "__main__":
    sys.exit(_QP.main(CFG, __doc__))
