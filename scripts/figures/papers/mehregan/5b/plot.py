#!/usr/bin/env python3
"""Mehregan et al. Figure 5b — post-train efficacy @ 30 Hz.

Thin entry point over ``mehregan/efficacy_panel.py`` (shared with Fig 5a).
Trains the 30 Hz model with the Fig 4a recipe ("all other parameters fixed",
§IV.A.2) and evaluates it closed loop next to PD no stim and periodic 30 Hz.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5b/plot.py --train
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5b/plot.py --plot-only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

from rl_adaptive_dbs.panel import load_script_module  # noqa: E402

_EFF = load_script_module(
    "mehregan_efficacy_panel", Path(__file__).resolve().parents[1] / "efficacy_panel.py"
)
CFG = _EFF.FIG5B


def fig5b_pass(panel: dict[str, Any]) -> dict[str, Any]:
    """Gate verdict from a manifest ``panel`` block (used by the gate-status table)."""
    return _EFF.efficacy_gates(CFG, panel)


if __name__ == "__main__":
    sys.exit(_EFF.main(CFG, __doc__))
