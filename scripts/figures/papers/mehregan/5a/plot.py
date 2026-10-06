#!/usr/bin/env python3
"""Mehregan et al. Figure 5a — post-train efficacy @ 45 Hz.

Thin entry point over ``mehregan/efficacy_panel.py`` (shared with Fig 5b).
Evaluates the Fig 4a 45 Hz checkpoint (the paper's one 45 Hz model) closed loop
next to PD no stim, periodic 45 Hz and periodic 130 Hz on the same seed.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5a/plot.py --plot-only
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
CFG = _EFF.FIG5A


def fig5a_pass(panel: dict[str, Any]) -> dict[str, Any]:
    """Gate verdict from a manifest ``panel`` block (used by the gate-status table)."""
    return _EFF.efficacy_gates(CFG, panel)


if __name__ == "__main__":
    sys.exit(_EFF.main(CFG, __doc__))
