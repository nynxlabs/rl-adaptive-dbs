#!/usr/bin/env python3
"""Materialize replication figure PNGs into the notes folder.

Scans each paper tracker ``docs/controllers/<controller>/README.md`` and its
figure pages ``docs/controllers/<controller>/figures/<panel>.md`` for image links
into ``docs/controllers/<controller>/figures/images/``, and copies the bytes to
the same repo-relative path under the notes folder (``RL_DBS_NOTES_DIR`` /
``export.notes_dir``, or detected from symlinked trackers — see
``rl_adaptive_dbs.notes_export``).

Images are addressed as ``(paper, rel)`` with ``rel`` relative to the paper's
``docs/controllers/<controller>/figures/`` folder (``images/<panel>/<name>.png``).

Plot scripts call this via ``promote.py`` when you pass ``--export-notes``; run manually anytime:

  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/export_notes_images.py
"""

from __future__ import annotations

from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs.panel import load_script_module

import argparse
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Any

_PROMOTE = Path(__file__).resolve().parent / "promote.py"
_promote = load_script_module("figure_promote", _PROMOTE)

REPO_ROOT = _promote.REPO_ROOT
PAPER_TRACKERS = doc_paths.all_trackers(REPO_ROOT)

_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_PAPERS = doc_paths.PAPERS
_SKIP_REPO_GLOB = ("paper.png", "/_full/")


def notes_root_dir() -> Path:
    """The notes folder (mirrors repo-relative paths)."""
    return _promote.require_notes_dir()


def _add_link(
    seen: set[tuple[str, str]],
    out: list[tuple[str, str]],
    paper: str,
    rel: str,
) -> None:
    if paper not in _PAPERS:
        return
    if rel.endswith("paper.png"):
        return
    if not rel.startswith("images/"):
        return
    key = (paper, rel)
    if key in seen:
        return
    seen.add(key)
    out.append(key)


def _doc_image_links(doc: Path, doc_rel: str) -> list[tuple[str, str]]:
    """``(paper, rel)`` for every image link in ``doc`` that lands in a figures folder."""
    out: list[tuple[str, str]] = []
    text = doc.read_text(encoding="utf-8")
    for match in _IMAGE_LINK_RE.finditer(text):
        href = match.group(1)
        if "://" in href or href.startswith("/"):
            continue
        split = doc_paths.split_figures_rel(doc_paths.resolve_doc_href(href, doc_rel))
        if split is not None:
            out.append(split)
    return out


def iter_tracker_image_links() -> list[tuple[str, str]]:
    """(paper, path under ``docs/controllers/<controller>/figures/``) from trackers + figure pages."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for paper in _PAPERS:
        tracker = doc_paths.tracker_path(paper, REPO_ROOT)
        docs = [tracker, *doc_paths.figure_pages(paper, REPO_ROOT)]
        for doc in docs:
            if not doc.is_file():
                continue
            doc_rel = doc_paths.rel_to_root(doc, REPO_ROOT) or doc.name
            for link_paper, rel in _doc_image_links(doc, doc_rel):
                _add_link(seen, out, link_paper, rel)
    return out


def iter_report3_image_links() -> list[tuple[str, str]]:
    """Image links in ``<notes_dir>/docs/reports/2026-08-10.md`` that point into figures folders."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        report_path = Path(__file__).resolve().parent / "update_report3.py"
        spec = importlib.util.spec_from_file_location("update_report3", report_path)
        if spec is None or spec.loader is None:
            return out
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        report_file = mod.report3_path()
    except FileNotFoundError:
        return out
    if not report_file.is_file():
        return out
    report_rel = (doc_paths.REPORTS_REL / report_file.name).as_posix()
    for paper, rel in _doc_image_links(report_file, report_rel):
        _add_link(seen, out, paper, rel)
    return out


def iter_materialized_repo_pngs() -> list[tuple[str, str]]:
    """Real (non-symlink) replication PNGs under ``docs/controllers/<controller>/figures/images/``."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for paper in _PAPERS:
        figures = doc_paths.figures_dir(paper, REPO_ROOT)
        images_dir = doc_paths.figure_images_dir(paper, root=REPO_ROOT)
        if not images_dir.is_dir():
            continue
        for path in images_dir.rglob("*.png"):
            if path.is_symlink():
                continue
            rel_path = str(path.relative_to(figures)).replace("\\", "/")
            if any(skip in rel_path for skip in _SKIP_REPO_GLOB):
                continue
            _add_link(seen, out, paper, rel_path)
    return out


def iter_all_push_targets(*, include_repo_materialized: bool = True) -> list[tuple[str, str]]:
    """Union of tracker/page links, the 2026-08-10 report gallery, and materialized repo PNGs."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for links in (iter_tracker_image_links(), iter_report3_image_links()):
        for paper, rel in links:
            _add_link(seen, out, paper, rel)
    if include_repo_materialized:
        for paper, rel in iter_materialized_repo_pngs():
            _add_link(seen, out, paper, rel)
    return out


def _resolve_source(paper: str, rel: str) -> Path | None:
    """Prefer materialized bytes in the active checkout, then main, then the notes folder.

    Panel plots often run from a ``.worktrees/`` checkout where ``REPO_ROOT`` is the
    main tree (for tracker promote) but the new ``_vN.png`` was written under the
    worktree. Search the worktree first when it differs from main.
    """
    candidates: list[Path] = []
    checkout = getattr(_promote, "CHECKOUT_ROOT", None)
    if isinstance(checkout, Path) and checkout.resolve() != REPO_ROOT.resolve():
        candidates.append(doc_paths.figures_dir(paper, checkout) / rel)
    candidates.append(doc_paths.figures_dir(paper, REPO_ROOT) / rel)
    for repo_path in candidates:
        if repo_path.is_file() and not repo_path.is_symlink():
            return repo_path
        if repo_path.is_symlink() and repo_path.is_file():
            return repo_path.resolve()
    notes_path = doc_paths.figures_dir(paper, notes_root_dir()) / rel
    if notes_path.is_file():
        return notes_path.resolve()
    for repo_path in candidates:
        resolved = repo_path.resolve()
        if resolved.is_file():
            return resolved
    return None


def push_image(
    paper: str,
    rel: str,
    *,
    notes_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy one replication PNG into the notes folder if source bytes differ.

    ``notes_root`` is the notes folder; the copy lands at the image's repo-relative
    path under it (``docs/controllers/<controller>/figures/<rel>``).
    """
    notes_root = notes_root or notes_root_dir()
    dest = doc_paths.figures_dir(paper, notes_root) / rel
    src = _resolve_source(paper, rel)
    result: dict[str, Any] = {
        "paper": paper,
        "rel": rel,
        "dest": str(dest),
        "action": "skip",
    }
    if src is None:
        result["action"] = "missing"
        return result
    result["src"] = str(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.is_symlink():
        dest.unlink()
    if dest.exists() and dest.samefile(src):
        result["action"] = "skip"
        return result
    if dest.exists():
        try:
            if dest.read_bytes() == src.read_bytes():
                result["action"] = "skip"
                return result
        except OSError:
            pass
    if dry_run:
        result["action"] = "would_copy"
        return result
    shutil.copy2(src, dest)
    result["action"] = "copied"
    return result


def push_all(
    *,
    dry_run: bool = False,
    verbose: bool = True,
    include_repo_materialized: bool = True,
    tracker_only: bool = False,
) -> dict[str, Any]:
    notes_root = notes_root_dir()
    results: list[dict[str, Any]] = []
    if tracker_only:
        targets = iter_tracker_image_links()
    else:
        targets = iter_all_push_targets(include_repo_materialized=include_repo_materialized)
    for paper, rel in targets:
        row = push_image(paper, rel, notes_root=notes_root, dry_run=dry_run)
        results.append(row)
        if verbose and row["action"] in {"copied", "would_copy", "missing"}:
            print(f"{row['action']}: {paper}/{rel}", flush=True)
    copied = sum(1 for r in results if r["action"] in {"copied", "would_copy"})
    missing = sum(1 for r in results if r["action"] == "missing")
    if verbose:
        print(
            f"export_notes: notes={notes_root} targets={len(targets)} "
            f"copied={copied} missing={missing}",
            flush=True,
        )
    return {
        "notes_root": str(notes_root),
        "n_links": len(results),
        "copied": copied,
        "missing": missing,
        "results": results,
    }


def push_repo_rel_png(repo_rel: str, *, dry_run: bool = False) -> None:
    """Push a single ``docs/controllers/<controller>/figures/images/...`` path, then refresh all links."""
    split = doc_paths.split_figures_rel(doc_paths.repo_rel_posix(repo_rel))
    if split is not None and split[1].startswith("images/"):
        push_image(split[0], split[1], dry_run=dry_run)
    push_all(dry_run=dry_run, verbose=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tracker-only",
        action="store_true",
        help="only tracker + figure page links (skip the 2026-08-10 report + materialized repo PNG scan)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    summary = push_all(
        dry_run=args.dry_run,
        verbose=not args.quiet,
        include_repo_materialized=not args.tracker_only,
        tracker_only=args.tracker_only,
    )
    if summary["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
