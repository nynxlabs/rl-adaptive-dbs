#!/usr/bin/env python3
"""Materialize replication figure PNGs into the knowledge-base vault.

Scans ``figures/<paper>/replications.md`` for ``![...](images/...)`` links and
copies bytes into the vault-backed ``figures/`` tree (``~/knowledge-base/...``).

Plot scripts call this via ``promote.py`` when you pass ``--push-kb``; run manually anytime:

  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/push_kb_images.py
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_PROMOTE = Path(__file__).resolve().parent / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_promote)

REPO_ROOT = _promote.REPO_ROOT
PAPER_TRACKERS = (
    _promote.resolve_paper_1_doc(),
    _promote.resolve_nguyen_doc(),
    _promote.resolve_ravivarapu_doc(),
)

_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\((images/[^)]+)\)")
_REPORT3_HREF_RE = re.compile(r"!\[[^\]]*\]\(\.\./figures/([^)]+)\)")
_PAPERS = ("mehregan", "nguyen", "ravivarapu")
_SKIP_REPO_GLOB = ("paper.png", "/_full/")


def kb_figures_root() -> Path:
    """Vault ``figures/`` directory (resolve through tracker symlinks)."""
    for doc in PAPER_TRACKERS:
        if doc.exists():
            return doc.resolve().parent.parent
    # Fallback: Insync vault layout on nynxbox.
    home = Path.home()
    for base in (home / "knowledge-base", home / "Insync" / "knowledge-base"):
        candidate = base / "bme" / "brain-stimulation" / "rl-adaptive-dbs" / "figures"
        if candidate.is_dir():
            return candidate
    msg = "could not locate vault figures/ root"
    raise FileNotFoundError(msg)


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


def iter_tracker_image_links() -> list[tuple[str, str]]:
    """(paper, repo-relative path under figures/<paper>/)."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in PAPER_TRACKERS:
        if not doc.is_file():
            continue
        paper = doc.resolve().parent.name
        if paper not in _PAPERS:
            continue
        text = doc.read_text(encoding="utf-8")
        for match in _IMAGE_LINK_RE.finditer(text):
            _add_link(seen, out, paper, match.group(1))
    return out


def iter_report3_image_links() -> list[tuple[str, str]]:
    """Gallery embeds from knowledge-base ``reports/3.md``."""
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
    for match in _REPORT3_HREF_RE.finditer(report_file.read_text(encoding="utf-8")):
        tail = match.group(1).strip()
        paper, _, rest = tail.partition("/")
        _add_link(seen, out, paper, rest)
    return out


def iter_materialized_repo_pngs() -> list[tuple[str, str]]:
    """Real (non-symlink) replication PNGs under ``figures/<paper>/images/``."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for paper in _PAPERS:
        images_dir = REPO_ROOT / "figures" / paper / "images"
        if not images_dir.is_dir():
            continue
        for path in images_dir.rglob("*.png"):
            if path.is_symlink():
                continue
            rel_path = str(path.relative_to(REPO_ROOT / "figures" / paper)).replace("\\", "/")
            if any(skip in rel_path for skip in _SKIP_REPO_GLOB):
                continue
            _add_link(seen, out, paper, rel_path)
    return out


def iter_all_push_targets(*, include_repo_materialized: bool = True) -> list[tuple[str, str]]:
    """Union of tracker links, Report 3 gallery, and materialized repo PNGs."""
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
    """Prefer materialized bytes in the active checkout, then main, then vault.

    Panel plots often run from a ``.worktrees/`` checkout where ``REPO_ROOT`` is the
    main tree (for tracker promote) but the new ``_vN.png`` was written under the
    worktree. Search the worktree first when it differs from main.
    """
    candidates: list[Path] = []
    checkout = getattr(_promote, "CHECKOUT_ROOT", None)
    if isinstance(checkout, Path) and checkout.resolve() != REPO_ROOT.resolve():
        candidates.append(checkout / "figures" / paper / rel)
    candidates.append(REPO_ROOT / "figures" / paper / rel)
    for repo_path in candidates:
        if repo_path.is_file() and not repo_path.is_symlink():
            return repo_path
        if repo_path.is_symlink() and repo_path.is_file():
            return repo_path.resolve()
    vault_path = kb_figures_root() / paper / rel
    if vault_path.is_file():
        return vault_path.resolve()
    for repo_path in candidates:
        resolved = repo_path.resolve()
        if resolved.is_file():
            return resolved
    return None


def push_image(
    paper: str,
    rel: str,
    *,
    vault_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy one replication PNG into the vault if source bytes differ."""
    vault_root = vault_root or kb_figures_root()
    dest = vault_root / paper / rel
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
    vault_root = kb_figures_root()
    results: list[dict[str, Any]] = []
    if tracker_only:
        targets = iter_tracker_image_links()
    else:
        targets = iter_all_push_targets(include_repo_materialized=include_repo_materialized)
    for paper, rel in targets:
        row = push_image(paper, rel, vault_root=vault_root, dry_run=dry_run)
        results.append(row)
        if verbose and row["action"] in {"copied", "would_copy", "missing"}:
            print(f"{row['action']}: {paper}/{rel}", flush=True)
    copied = sum(1 for r in results if r["action"] in {"copied", "would_copy"})
    missing = sum(1 for r in results if r["action"] == "missing")
    if verbose:
        print(
            f"push_kb: vault={vault_root} targets={len(targets)} "
            f"copied={copied} missing={missing}",
            flush=True,
        )
    return {
        "vault_root": str(vault_root),
        "n_links": len(results),
        "copied": copied,
        "missing": missing,
        "results": results,
    }


def push_repo_rel_png(repo_rel: str, *, dry_run: bool = False) -> None:
    """Push a single ``figures/<paper>/images/...`` path, then refresh all links."""
    text = repo_rel.replace("\\", "/")
    marker = "figures/"
    idx = text.find(marker)
    tail = text[idx + len(marker) :] if idx >= 0 else text
    paper, _, rest = tail.partition("/")
    if paper in _PAPERS and rest.startswith("images/"):
        push_image(paper, rest, dry_run=dry_run)
    push_all(dry_run=dry_run, verbose=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tracker-only",
        action="store_true",
        help="only ``replications.md`` links (skip Report 3 + materialized repo PNG scan)",
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
