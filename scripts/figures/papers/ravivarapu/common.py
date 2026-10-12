"""Shared helpers for Ravivarapu panel scripts (PNG placement, promote switch)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

FIG4_CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")


def add_no_promote_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help=(
            "Write the PNG and manifest locally only: skip the tracker promote, the "
            "notes-folder PNG link, --export-notes, and --update-report (worktree iteration)"
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

    With ``promote`` the PNG lands next to the main checkout's ``paper.png``
    symlink target (the notes folder), so tracker and gallery links resolve.
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
        paper = root / path.parent / "paper.png"
        if not paper.is_symlink():
            continue
        linked_dir = paper.resolve().parent
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
