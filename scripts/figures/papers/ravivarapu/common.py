"""Shared helpers for Ravivarapu panel scripts (PNG placement, promote switch)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rl_adaptive_dbs.panel import load_script_module

_LINKED = load_script_module("figure_linked", Path(__file__).resolve().parents[1] / "_linked.py")

FIG4_CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")


def add_no_promote_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help=(
            "Write the PNG and manifest locally only: skip the docs promote, the "
            "notes-folder PNG link, and --export-notes (worktree iteration)"
        ),
    )


def fig4_checkpoint(variant: str, seed: int) -> Path:
    """Fig 4 training checkpoint shared by Figs 4–7 (one config for every panel)."""
    return FIG4_CACHE_DIR / f"{variant}_train{seed}.pt"


def rolling_mean(values: Any, window: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if window <= 1 or y.size == 0:
        return y
    out = np.empty_like(y)
    for i in range(y.size):
        lo = max(0, i - window + 1)
        out[i] = float(y[lo : i + 1].mean())
    return out


def linked_png(path: Path, promote_module: Any, *, promote: bool = True) -> Path:
    """Where ``savefig`` should write a versioned replication PNG.

    With ``promote`` the PNG lands in the notes folder the main checkout's
    panel folder links into (``paper.png`` or another symlinked PNG there;
    ``papers/_linked.py``), so figure page and gallery links resolve.
    Without it the PNG stays a plain local file (worktree iteration).
    """
    path = Path(path)
    if not promote:
        local = path if path.is_absolute() else Path.cwd() / path
        local.parent.mkdir(parents=True, exist_ok=True)
        return local
    roots: list[Path] = []
    main_root = getattr(promote_module, "REPO_ROOT", None)
    if isinstance(main_root, Path):
        roots.append(main_root)
    roots.append(Path.cwd())
    for root in roots:
        linked_dir = _LINKED.linked_dir(root / path.parent)
        if linked_dir is None:
            # Panels without a linked PNG yet (Table II): map through a
            # sibling panel's link to the same notes images folder.
            images = root / path.parent.parent
            siblings = sorted(p for p in images.iterdir() if p.is_dir()) if images.is_dir() else []
            linked = [d for d in map(_LINKED.linked_dir, siblings) if d is not None]
            if not linked:
                continue
            linked_dir = linked[0].parent / path.parent.name
        linked_target = linked_dir / path.name
        local = path if path.is_absolute() else Path.cwd() / path
        local.parent.mkdir(parents=True, exist_ok=True)
        if not linked_target.exists():
            linked_target.parent.mkdir(parents=True, exist_ok=True)
            linked_target.touch()
        if local.exists() or local.is_symlink():
            if local.resolve() != linked_target.resolve():
                return linked_target
            return local
        try:
            local.symlink_to(linked_target)
        except OSError:
            return linked_target
        return local
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
