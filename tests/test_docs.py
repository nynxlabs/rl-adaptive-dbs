"""Docs layout: plant.md presence, sections, and resolvable markdown links."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
PLANT_MD = DOCS / "plant.md"

# Headings required in plant.md (stable contract for the spec).
PLANT_REQUIRED_SECTIONS = (
    "# Plant specification",
    "## 1. Scope",
    "## 2. Reference and provenance",
    "## 3. Network topology and dynamics",
    "## 4. STN DBS actuation",
    "## 5. Integration and simulated time",
    "## 6. Biomarkers from GPi spiking",
    "## 7. Target plant wrapper API",
    "## 8. Equivalence and validation",
    "## 11. Consistency checklist",
)

# [text](target) — exclude http(s) and mailto
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _markdown_link_targets(markdown_path: Path) -> list[tuple[str, Path]]:
    text = markdown_path.read_text(encoding="utf-8")
    base = markdown_path.parent
    out: list[tuple[str, Path]] = []
    for raw in LINK_RE.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (base / target).resolve()
        out.append((target, resolved))
    return out


@pytest.mark.parametrize("heading", PLANT_REQUIRED_SECTIONS)
def test_plant_md_has_required_sections(heading: str) -> None:
    assert PLANT_MD.is_file(), "docs/plant.md must exist"
    content = PLANT_MD.read_text(encoding="utf-8")
    assert heading in content, f"missing section: {heading}"


def test_plant_md_links_resolve() -> None:
    missing: list[str] = []
    for target, resolved in _markdown_link_targets(PLANT_MD):
        if not resolved.exists():
            missing.append(f"{target} -> {resolved}")
    assert not missing, "broken links in plant.md:\n" + "\n".join(missing)


@pytest.mark.parametrize(
    "doc_path",
    [
        DOCS / "environment.md",
        DOCS / "development.md",
        DOCS / "development" / "roadmap.md",
        DOCS / "benchmarking.md",
        DOCS / "getting_started.md",
        DOCS / "testing.md",
        REPO_ROOT / "README.md",
    ],
    ids=lambda p: p.name,
)
def test_docs_linking_to_plant_resolve(doc_path: Path) -> None:
    """Files that reference plant.md must use a resolvable relative link."""
    if not doc_path.is_file():
        pytest.skip(f"{doc_path} not present")
    text = doc_path.read_text(encoding="utf-8")
    if "plant.md" not in text:
        return
    for target, resolved in _markdown_link_targets(doc_path):
        if target.endswith("plant.md") or target.endswith("/plant.md"):
            assert resolved == PLANT_MD.resolve(), f"{doc_path.name}: plant link points to {resolved}"
            assert PLANT_MD.is_file()
