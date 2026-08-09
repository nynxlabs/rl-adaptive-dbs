"""Load ``paper_overlay`` without package-relative imports in panel plot scripts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_paper_overlay() -> ModuleType:
    overlay_path = Path(__file__).resolve().parent / "paper_overlay.py"
    spec = importlib.util.spec_from_file_location("figure_paper_overlay", overlay_path)
    if spec is None or spec.loader is None:
        msg = f"cannot load paper overlay module from {overlay_path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
