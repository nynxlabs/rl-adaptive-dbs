"""Follow a panel's images folder into a linked notes folder, when it has one.

A panel folder ``docs/controllers/<controller>/figures/images/<panel>/`` can hold symlinks into an
external notes folder. New versioned replication PNGs should land there too,
so the notes and the repo see the same file. The marker is ``paper.png`` when
it is a symlink; ``paper.png`` is a local-only crop that may be missing or a
plain file (``scripts/digitization/paper_crops.py``), so any other symlinked
``*.png`` in the folder counts as well. With neither, the folder is plain.
"""
from __future__ import annotations

from pathlib import Path


def linked_dir(folder: Path, *, exclude: str | None = None) -> Path | None:
    """The folder that ``folder``'s symlinked PNGs point into, else ``None``."""
    folder = Path(folder)
    paper = folder / "paper.png"
    if paper.is_symlink():
        return paper.resolve().parent
    if not folder.is_dir():
        return None
    here = folder.resolve()
    for png in sorted(folder.glob("*.png")):
        if png.name == exclude or not png.is_symlink():
            continue
        target = png.resolve().parent
        if target != here:
            return target
    return None


def linked_png(path: Path) -> Path:
    """Where ``savefig`` should write ``path``.

    In a linked folder a new ``path`` becomes a symlink to the same name in the
    notes folder (the target is created empty, then written through the link).
    Otherwise, or when ``path`` already exists, it is returned unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    target_dir = linked_dir(path.parent, exclude=path.name)
    if target_dir is None:
        return path
    target = target_dir / path.name
    if not target.exists():
        target.touch()
    path.symlink_to(target)
    return path
