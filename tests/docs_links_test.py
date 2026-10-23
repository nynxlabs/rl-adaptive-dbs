"""Every relative link and image in the public markdown resolves to a real file."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Public-facing docs.
PUBLIC_MARKDOWN = sorted([REPO_ROOT / "README.md"] + list((REPO_ROOT / "docs").rglob("*.md")))

FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# [text](target) and ![alt](target "title")
MD_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# <img src="target">
HTML_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=\"([^\"]+)\"", re.IGNORECASE)
# <source srcset="target"> (the dark variant in a light/dark <picture> pair)
HTML_SRCSET_RE = re.compile(r"<source\b[^>]*\bsrcset=\"([^\"]+)\"", re.IGNORECASE)
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _relative_targets(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    text = INLINE_CODE_RE.sub("", FENCED_CODE_RE.sub("", text))
    targets = MD_LINK_RE.findall(text) + HTML_SRC_RE.findall(text) + HTML_SRCSET_RE.findall(text)
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


# A replication or paper-redraw image: light/dark pair or a plain one-theme image.
PICTURE_RE = re.compile(
    r'<picture><source media="\(prefers-color-scheme: dark\)" srcset="([^"]+)">'
    r'<img src="([^"]+)" alt="([^"]*)"[^>]*></picture>'
)
PLAIN_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+\.png)\)|<img src="([^"]+\.png)"[^>]*alt="([^"]*)"')
_PAIRED_ALT = re.compile(r"^(Replication|Digitized from)|\breplication$")


@pytest.mark.parametrize(
    "markdown_path",
    PUBLIC_MARKDOWN,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_replication_images_are_light_dark_pairs(markdown_path: Path) -> None:
    """Replication and redraw PNGs show as a ``<picture>`` with their ``_dark`` variant."""
    text = INLINE_CODE_RE.sub("", FENCED_CODE_RE.sub("", markdown_path.read_text(encoding="utf-8")))
    for dark, light, _alt in PICTURE_RE.findall(text):
        assert dark == light[: -len(".png")] + "_dark.png", f"{markdown_path}: {dark} is not {light}'s dark variant"
    single = []
    for match in PLAIN_IMAGE_RE.finditer(PICTURE_RE.sub("", text)):
        alt = match.group(1) if match.group(1) is not None else match.group(4)
        if _PAIRED_ALT.search(alt or ""):
            single.append(match.group(0))
    assert not single, f"single-theme replication images in {markdown_path.relative_to(REPO_ROOT)}: {single}"
