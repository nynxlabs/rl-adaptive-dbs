"""Paper crop spec (``paper_crops.json``) and the linked-PNG helper."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "digitization"))

import paper_crops  # noqa: E402

from rl_adaptive_dbs.panel import load_script_module  # noqa: E402

_LINKED = load_script_module("figure_linked", REPO / "scripts" / "figures" / "papers" / "_linked.py")

PUBLISHER_PANELS = {
    *(f"mehregan/{p}" for p in ("1b", "2a", "2b", "4a", "4b", "5a", "5b", "6a", "6b")),
    *(f"nguyen/{p}" for p in ("3", "4", "5", "6", "7")),
    *(f"ravivarapu/{p}" for p in ("4a", "4b", "5a", "5b", "6", "7")),
}


def test_spec_covers_every_panel_with_consistent_geometry():
    spec = paper_crops.load_spec()
    pub = spec["sources"]["publisher"]
    assert set(pub["panels"]) == PUBLISHER_PANELS
    for source in spec["sources"].values():
        for panel, entry in source["panels"].items():
            assert paper_crops.panel_paper(panel) in source["papers"], panel
            scale, x, y, w, h = paper_crops.panel_geometry(entry)
            assert 1.0 <= scale <= 5.0, panel
            assert x >= 0 and y >= 0 and [w, h] == entry["size_px"], panel


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_publisher_crops_are_gitignored_and_arxiv_crops_are_not():
    spec = paper_crops.load_spec()

    def ignored(rel: str) -> bool:
        res = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
        return res.returncode == 0

    for panel in spec["sources"]["publisher"]["panels"]:
        rel = paper_crops.output_path(Path("."), panel, "paper.png").as_posix()
        assert ignored(rel), rel
    for panel in spec["sources"]["arxiv"]["panels"]:
        rel = paper_crops.output_path(Path("."), panel, spec["sources"]["arxiv"]["output"]).as_posix()
        assert not ignored(rel), rel


def test_missing_pdf_names_the_paper_and_folders(tmp_path):
    source = paper_crops.load_spec()["sources"]["publisher"]
    with pytest.raises(paper_crops.CropError, match="mehregan.pdf"):
        paper_crops.find_pdf(source, "mehregan", [tmp_path])


def test_pdf_lookup_accepts_alias_names(tmp_path):
    source = paper_crops.load_spec()["sources"]["publisher"]
    (tmp_path / "paper_2.pdf").write_bytes(b"%PDF-1.4\n")
    assert paper_crops.find_pdf(source, "nguyen", [tmp_path / "missing", tmp_path]).name == "paper_2.pdf"


def _can_symlink(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe").symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    return True


def test_linked_png_follows_any_symlinked_png(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks unavailable")
    notes, panel = tmp_path / "notes" / "4a", tmp_path / "repo" / "4a"
    notes.mkdir(parents=True)
    panel.mkdir(parents=True)
    (notes / "old_v1.png").write_bytes(b"")
    (panel / "old_v1.png").symlink_to(notes / "old_v1.png")  # no paper.png in the folder
    out = _LINKED.linked_png(panel / "new_v2.png")
    assert out.is_symlink() and out.resolve() == (notes / "new_v2.png").resolve()


def test_linked_png_plain_folder(tmp_path):
    out = _LINKED.linked_png(tmp_path / "4a" / "fig_v1.png")
    assert out == tmp_path / "4a" / "fig_v1.png" and not out.is_symlink()
    assert out.parent.is_dir() and not os.listdir(out.parent)
