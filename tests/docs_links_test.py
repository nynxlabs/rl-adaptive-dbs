"""Every relative link and image in the public markdown resolves to a real file."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Public-facing docs. reports/ holds frozen copies of sent reports and is excluded.
PUBLIC_MARKDOWN = sorted(
    [REPO_ROOT / "README.md"]
    + list((REPO_ROOT / "docs").rglob("*.md"))
    + list((REPO_ROOT / "figures").rglob("*.md"))
)

FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# [text](target) and ![alt](target "title")
MD_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# <img src="target">
HTML_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=\"([^\"]+)\"", re.IGNORECASE)
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _relative_targets(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    text = INLINE_CODE_RE.sub("", FENCED_CODE_RE.sub("", text))
    targets = MD_LINK_RE.findall(text) + HTML_SRC_RE.findall(text)
    out = []
    for raw in targets:
        target = raw.split("#", 1)[0].strip()
        if target and not SCHEME_RE.match(target):
            out.append(target)
    return out


@pytest.mark.parametrize(
    "markdown_path",
    PUBLIC_MARKDOWN,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_relative_links_resolve(markdown_path: Path) -> None:
    missing = [
        target
        for target in _relative_targets(markdown_path)
        if not (markdown_path.parent / target).exists()
    ]
    assert not missing, f"Broken links in {markdown_path.relative_to(REPO_ROOT)}: {missing}"
