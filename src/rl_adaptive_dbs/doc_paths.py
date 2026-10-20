"""Where the paper-replication docs and figure images live in the repo.

Each replicated paper maps to one controller folder under ``docs/controllers/``:

- ``docs/controllers/<controller>/README.md`` — the paper's **tracker**: the
  ``<!-- summary:start/end -->`` block plus the panel table.
- ``docs/controllers/<controller>/figures/<panel>.md`` — one **figure page**
  per panel, holding that panel's ``caption-<id>`` / ``gates-<id>`` marker
  blocks and its replication / paper image links. ``<panel>`` is the plot
  script folder under ``scripts/figures/papers/<paper>/`` (``1b``, ``4``,
  ``table2``, …).
- ``docs/controllers/<controller>/figures/images/<panel>/`` — that panel's
  images. Links from a figure page are ``images/<panel>/…``; from the tracker
  they are ``figures/images/<panel>/…``.

A notes folder (``rl_adaptive_dbs.notes_export``) mirrors these repo-relative
paths. Everything here is plain path arithmetic — no heavy imports.
"""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]

PAPER_CONTROLLER: dict[str, str] = {
    "mehregan": "ddpg",
    "nguyen": "snn",
    "ravivarapu": "sea_dbs",
}
CONTROLLER_PAPER: dict[str, str] = {ctrl: paper for paper, ctrl in PAPER_CONTROLLER.items()}
PAPERS: tuple[str, ...] = tuple(PAPER_CONTROLLER)

# Marker ids that differ from the panel (plot script folder) name.
MARKER_PANEL_ALIASES: dict[str, dict[str, str]] = {
    "ravivarapu": {"t2": "table2"},
}

CONTROLLERS_DOCS_REL = PurePosixPath("docs/controllers")


def controller_for(paper: str) -> str:
    """Controller folder name for ``paper`` (``mehregan`` → ``ddpg``)."""
    try:
        return PAPER_CONTROLLER[paper]
    except KeyError:
        msg = f"unknown paper {paper!r}; expected one of {sorted(PAPER_CONTROLLER)}"
        raise KeyError(msg) from None


def panel_for_marker_id(paper: str, marker_id: str) -> str:
    """Panel (figure page name) for a ``caption-<id>`` / ``gates-<id>`` marker id."""
    return MARKER_PANEL_ALIASES.get(paper, {}).get(marker_id, marker_id)


def controller_docs_rel(paper: str) -> PurePosixPath:
    """``docs/controllers/<controller>`` (repo-relative)."""
    return CONTROLLERS_DOCS_REL / controller_for(paper)


def figures_rel(paper: str) -> PurePosixPath:
    """``docs/controllers/<controller>/figures`` (repo-relative)."""
    return controller_docs_rel(paper) / "figures"


def tracker_rel(paper: str) -> PurePosixPath:
    return controller_docs_rel(paper) / "README.md"


def figure_page_rel(paper: str, panel: str) -> PurePosixPath:
    return figures_rel(paper) / f"{panel}.md"


def figure_images_rel(paper: str, panel: str | None = None) -> PurePosixPath:
    """``docs/controllers/<controller>/figures/images[/<panel>]`` (repo-relative)."""
    base = figures_rel(paper) / "images"
    return base / panel if panel else base


def controller_docs_dir(paper: str, root: Path = REPO) -> Path:
    return Path(root) / controller_docs_rel(paper)


def figures_dir(paper: str, root: Path = REPO) -> Path:
    return Path(root) / figures_rel(paper)


def tracker_path(paper: str, root: Path = REPO) -> Path:
    """The paper tracker ``docs/controllers/<controller>/README.md``."""
    return Path(root) / tracker_rel(paper)


def figure_page(paper: str, panel: str, root: Path = REPO) -> Path:
    """The panel page ``docs/controllers/<controller>/figures/<panel>.md``."""
    return Path(root) / figure_page_rel(paper, panel)


def figure_images_dir(paper: str, panel: str | None = None, root: Path = REPO) -> Path:
    """``docs/controllers/<controller>/figures/images[/<panel>]``."""
    return Path(root) / figure_images_rel(paper, panel)


def tracker_figure_page(tracker: Path, panel: str) -> Path:
    """The ``figures/<panel>.md`` page beside a tracker ``README.md``."""
    return Path(tracker).parent / "figures" / f"{panel}.md"


def figure_pages(paper: str, root: Path = REPO) -> list[Path]:
    """Existing figure pages for ``paper`` (sorted)."""
    folder = figures_dir(paper, root)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.md") if p.is_file())


def all_trackers(root: Path = REPO) -> tuple[Path, ...]:
    """Every paper tracker path (existing or not), in ``PAPERS`` order."""
    return tuple(tracker_path(paper, root) for paper in PAPERS)


def main_checkout_root(checkout: Path) -> Path:
    """The main repo checkout when ``checkout`` is ``.../.worktrees/<name>``."""
    resolved = Path(checkout).resolve()
    parts = resolved.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        if idx > 0:
            return Path(*parts[:idx])
    return resolved


def repo_rel_posix(path: Path | str, root: Path | None = None) -> str:
    """Repo-relative POSIX form of a docs/artifacts path.

    Accepts absolute paths under ``root`` (or any checkout of this repo, by
    finding ``docs/`` / ``artifacts/`` in the path) and already-relative paths.
    """
    text = Path(path).as_posix() if not isinstance(path, str) else path.replace("\\", "/")
    if root is not None:
        try:
            return Path(text).relative_to(Path(root)).as_posix()
        except ValueError:
            pass
    for marker in ("docs/controllers/", "artifacts/", "docs/"):
        if text.startswith(marker):
            return text
        idx = text.rfind(f"/{marker}")
        if idx >= 0:
            return text[idx + 1 :]
    return text


def split_figures_rel(repo_rel: str) -> tuple[str, str] | None:
    """``docs/controllers/<ctrl>/figures/<rest>`` → ``(paper, <rest>)``, else ``None``."""
    parts = PurePosixPath(repo_rel.replace("\\", "/")).parts
    if len(parts) < 5 or parts[0] != "docs" or parts[1] != "controllers" or parts[3] != "figures":
        return None
    paper = CONTROLLER_PAPER.get(parts[2])
    if paper is None:
        return None
    return paper, "/".join(parts[4:])


def doc_link(target_rel: str, doc_rel: str | PurePosixPath) -> str:
    """Markdown href from repo-relative doc ``doc_rel`` to repo-relative ``target_rel``."""
    start = posixpath.dirname(PurePosixPath(doc_rel).as_posix()) or "."
    return posixpath.relpath(target_rel.replace("\\", "/"), start)


def resolve_doc_href(href: str, doc_rel: str | PurePosixPath) -> str:
    """Repo-relative target of ``href`` written in repo-relative doc ``doc_rel``."""
    base = posixpath.dirname(PurePosixPath(doc_rel).as_posix())
    return posixpath.normpath(posixpath.join(base, href))


def rel_to_root(path: Path, root: Path) -> str | None:
    """``path`` relative to ``root`` (no symlink resolution), else ``None``."""
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return None


def _marker_re(marker: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- {re.escape(marker)}:start -->)(.*?)(<!-- {re.escape(marker)}:end -->)",
        re.DOTALL,
    )


class DocSet:
    """Edit a few markdown docs at once, writing each one only if it changed.

    ``paths`` is the search order — normally ``[figure page, tracker]`` so a
    panel's marker blocks and image links are written in its page, with the
    tracker as the fallback while a block has not moved yet. Missing files are
    skipped. Every edit goes to the first doc that holds its target.
    """

    def __init__(self, paths: Iterable[Path], *, root: Path | None = None) -> None:
        self.paths = [Path(p) for p in paths]
        self.root = Path(root) if root is not None else None
        self._text: dict[Path, str] = {}
        self._orig: dict[Path, str] = {}

    @classmethod
    def for_panel(cls, tracker: Path, panel: str, *, root: Path | None = None) -> "DocSet":
        """``[figures/<panel>.md, README.md]`` for the tracker at ``tracker``."""
        return cls([tracker_figure_page(tracker, panel), tracker], root=root)

    def view(self, paths: Iterable[Path]) -> "DocSet":
        """Another search order over the same in-memory docs (edits are shared)."""
        other = DocSet(paths, root=self.root)
        other._text = self._text
        other._orig = self._orig
        return other

    def panel_view(self, tracker: Path, panel: str) -> "DocSet":
        """``view`` over ``[figures/<panel>.md, tracker]``."""
        return self.view([tracker_figure_page(tracker, panel), tracker])

    def exists(self) -> bool:
        return any(p.is_file() for p in self.paths)

    def text(self, path: Path) -> str | None:
        path = Path(path)
        if path not in self._text:
            if not path.is_file():
                return None
            raw = path.read_text(encoding="utf-8")
            self._text[path] = raw
            self._orig[path] = raw
        return self._text[path]

    def set_text(self, path: Path, text: str) -> None:
        path = Path(path)
        if self.text(path) is None:
            self._orig[path] = ""
        self._text[path] = text

    def _where(self, pred: Callable[[str], bool]) -> Path | None:
        for path in self.paths:
            text = self.text(path)
            if text is not None and pred(text):
                return path
        return None

    def find(self, needle: str) -> Path | None:
        """First doc whose text contains ``needle``."""
        return self._where(lambda text: needle in text)

    def search(self, pattern: str | re.Pattern[str]) -> Path | None:
        """First doc where ``pattern`` matches."""
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        return self._where(lambda text: regex.search(text) is not None)

    def marker_doc(self, marker: str) -> Path | None:
        return self._where(lambda text: _marker_re(marker).search(text) is not None)

    def has_marker(self, marker: str) -> bool:
        return self.marker_doc(marker) is not None

    def missing_label(self) -> str:
        return " or ".join(str(p) for p in self.paths)

    def replace_marker(self, marker: str, body: str, *, error: str | None = None) -> Path:
        """Set the body of ``<!-- marker:start/end -->`` in the first doc that has it."""
        path = self.marker_doc(marker)
        if path is None:
            raise ValueError(error or f"missing marker block {marker} in {self.missing_label()}")
        text = self._text[path]
        self._text[path] = _marker_re(marker).sub(
            lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text, count=1
        )
        return path

    def read_marker(self, marker: str) -> str | None:
        path = self.marker_doc(marker)
        if path is None:
            return None
        match = _marker_re(marker).search(self._text[path])
        return match.group(2) if match else None

    def sub(
        self,
        pattern: str | re.Pattern[str],
        repl: str | Callable[[re.Match[str]], str],
        *,
        count: int = 0,
        path: Path | None = None,
    ) -> Path | None:
        """``re.sub`` in ``path`` (or the first doc where ``pattern`` matches)."""
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        target = Path(path) if path is not None else self.search(regex)
        if target is None or self.text(target) is None:
            return None
        self._text[target] = regex.sub(repl, self._text[target], count=count)
        return target

    def replace(self, old: str, new: str | Callable[[Path], str], *, count: int = -1) -> Path | None:
        """``str.replace`` in the first doc containing ``old``; ``new`` may depend on the doc."""
        target = self.find(old)
        if target is None:
            return None
        value = new(target) if callable(new) else new
        self._text[target] = self._text[target].replace(old, value, count)
        return target

    def href(self, target_rel: str, doc: Path) -> str:
        """Markdown href from ``doc`` to repo-relative ``target_rel``.

        Uses ``doc``'s path under ``root`` (unresolved, so symlinked docs link the
        way they do in the repo); falls back to the doc's place under
        ``docs/controllers/<controller>/``.
        """
        doc_rel = rel_to_root(doc, self.root) if self.root is not None else None
        if doc_rel is None:
            doc_rel = _doc_rel_from_tail(Path(doc))
        return doc_link(target_rel, doc_rel)

    def changed(self) -> list[Path]:
        return [p for p, t in self._text.items() if t != self._orig.get(p)]

    def save(self) -> list[Path]:
        """Write every changed doc; returns the paths written."""
        written = self.changed()
        for path in written:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._text[path], encoding="utf-8")
            self._orig[path] = self._text[path]
        return written


def _doc_rel_from_tail(doc: Path) -> str:
    """Repo-relative path of a doc from its ``docs/controllers/<ctrl>/…`` tail."""
    parts = doc.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "docs" and i + 1 < len(parts) and parts[i + 1] == "controllers":
            return "/".join(parts[i:])
    return doc.name
