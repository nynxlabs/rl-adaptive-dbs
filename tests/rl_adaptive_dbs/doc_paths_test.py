"""Paper doc paths (``rl_adaptive_dbs.doc_paths``) and promote's page-first doc edits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs.doc_paths import DocSet
from rl_adaptive_dbs.panel import load_script_module

REPO = Path(__file__).resolve().parents[2]


def test_layout_paths(tmp_path: Path) -> None:
    assert doc_paths.tracker_path("mehregan", tmp_path) == tmp_path / "docs/controllers/ddpg/README.md"
    assert doc_paths.figure_page("ravivarapu", "table2", tmp_path) == (
        tmp_path / "docs/controllers/sea_dbs/figures/table2.md"
    )
    assert doc_paths.figure_images_dir("nguyen", "4", tmp_path) == (
        tmp_path / "docs/controllers/snn/figures/images/4"
    )
    assert doc_paths.all_trackers(tmp_path) == tuple(
        tmp_path / "docs/controllers" / ctrl / "README.md" for ctrl in ("ddpg", "snn", "sea_dbs")
    )
    assert doc_paths.panel_for_marker_id("ravivarapu", "t2") == "table2"
    assert doc_paths.panel_for_marker_id("nguyen", "4") == "4"
    with pytest.raises(KeyError, match="unknown paper"):
        doc_paths.controller_for("nobody")


def test_hrefs_are_relative_to_each_doc() -> None:
    png = "docs/controllers/ddpg/figures/images/1b/gpi_psd_v3.png"
    assert doc_paths.doc_link(png, "docs/controllers/ddpg/figures/1b.md") == "images/1b/gpi_psd_v3.png"
    assert doc_paths.doc_link(png, "docs/controllers/ddpg/README.md") == "figures/images/1b/gpi_psd_v3.png"
    assert doc_paths.doc_link(png, "docs/guides/cli.md") == (
        "../controllers/ddpg/figures/images/1b/gpi_psd_v3.png"
    )
    assert doc_paths.resolve_doc_href("images/1b/x.png", "docs/controllers/ddpg/figures/1b.md") == (
        "docs/controllers/ddpg/figures/images/1b/x.png"
    )
    assert doc_paths.split_figures_rel("docs/controllers/snn/figures/images/4/x.png") == (
        "nguyen",
        "images/4/x.png",
    )
    assert doc_paths.split_figures_rel("artifacts/figures/papers/nguyen/4/x.png") is None
    assert doc_paths.repo_rel_posix("/x/repo/.worktrees/t/docs/controllers/snn/figures/images/4/a.png") == (
        "docs/controllers/snn/figures/images/4/a.png"
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_docset_prefers_page_and_falls_back_to_tracker(tmp_path: Path) -> None:
    tracker = _write(
        doc_paths.tracker_path("nguyen", tmp_path),
        "<!-- summary:start -->\nold\n<!-- summary:end -->\n"
        "<!-- caption-5:start -->\nold5\n<!-- caption-5:end -->\n",
    )
    page = _write(
        doc_paths.figure_page("nguyen", "4", tmp_path),
        "<!-- caption-4:start -->\nold4\n<!-- caption-4:end -->\n",
    )
    docs = DocSet([tracker], root=tmp_path)
    assert docs.panel_view(tracker, "4").replace_marker("caption-4", "new4") == page
    assert docs.panel_view(tracker, "5").replace_marker("caption-5", "new5") == tracker
    with pytest.raises(ValueError, match="missing marker block caption-6"):
        docs.panel_view(tracker, "6").replace_marker("caption-6", "x")
    assert sorted(docs.save()) == sorted([page, tracker])
    assert "new4" in page.read_text() and "new5" in tracker.read_text()
    assert docs.save() == []  # nothing changed since


@pytest.fixture
def promote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = load_script_module("figure_promote_doc_paths_test", REPO / "scripts/figures/papers/promote.py")
    sys.modules.setdefault("figure_promote_doc_paths_test", mod)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CHECKOUT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "PAPER_RAVIVARAPU_DOC", doc_paths.tracker_path("ravivarapu", tmp_path))
    monkeypatch.setattr(mod, "PAPER_NGUYEN_DOC", doc_paths.tracker_path("nguyen", tmp_path))
    monkeypatch.setattr(mod, "PAPER_1_DOC", doc_paths.tracker_path("mehregan", tmp_path))
    monkeypatch.setattr(mod, "refresh_ravivarapu_gate_tables", lambda **_: None)
    monkeypatch.setattr(mod, "refresh_mehregan_gate_tables", lambda **_: None)
    monkeypatch.setattr(mod, "materialize_ship_png", lambda *a, **k: None)
    return mod


_PANEL_BLOCK = (
    "![Replication Fig 5a](images/5a/inference_50hz_v1.png)\n\n"
    "<!-- caption-5a:start -->\nold\n<!-- caption-5a:end -->\n\n"
    "**Status:** Fail (rep v1).\n"
)


def test_promote_writes_figure_page(promote, tmp_path: Path) -> None:
    tracker = _write(doc_paths.tracker_path("ravivarapu", tmp_path), "# tracker\n")
    page = _write(doc_paths.figure_page("ravivarapu", "5a", tmp_path), _PANEL_BLOCK)
    png = doc_paths.figure_images_dir("ravivarapu", "5a", tmp_path) / "inference_50hz_v2.png"
    manifest = {"caption": "c", "png_version": 2, "gates": {"pass": True}, "carrier_hz": 50}
    promote.promote_ravivarapu_5a(manifest=manifest, png_path=png)
    text = page.read_text()
    assert "![Replication Fig 5a](images/5a/inference_50hz_v2.png)" in text
    assert "**Caption:** c (v2)" in text
    assert "**Status:** **Pass** (rep v2)" in text
    assert tracker.read_text() == "# tracker\n"


def test_promote_falls_back_to_tracker(promote, tmp_path: Path) -> None:
    tracker_block = _PANEL_BLOCK.replace("(images/", "(figures/images/")
    tracker = _write(doc_paths.tracker_path("ravivarapu", tmp_path), tracker_block)
    _write(doc_paths.figure_page("ravivarapu", "5a", tmp_path), "# Fig 5a page, blocks not moved yet\n")
    png = doc_paths.figure_images_dir("ravivarapu", "5a", tmp_path) / "inference_50hz_v2.png"
    promote.promote_ravivarapu_5a(manifest={"caption": "c", "png_version": 2}, png_path=png)
    text = tracker.read_text()
    assert "![Replication Fig 5a](figures/images/5a/inference_50hz_v2.png)" in text
    assert "**Caption:** c (v2)" in text


def test_promote_raises_when_no_doc_has_the_marker(promote, tmp_path: Path) -> None:
    _write(doc_paths.tracker_path("nguyen", tmp_path), "<!-- summary:start -->\n<!-- summary:end -->\n")
    _write(doc_paths.figure_page("nguyen", "5", tmp_path), "# Fig 5\n")
    png = doc_paths.figure_images_dir("nguyen", "5", tmp_path) / "spikes_energy_v2.png"
    with pytest.raises(ValueError, match="missing marker block caption-5"):
        promote.promote_nguyen_5(manifest={"caption": "c"}, png_path=png)


def test_nguyen_promote_splits_page_and_tracker(promote, tmp_path: Path) -> None:
    tracker = _write(
        doc_paths.tracker_path("nguyen", tmp_path),
        "<!-- summary:start -->\nold\n<!-- summary:end -->\n",
    )
    page = _write(
        doc_paths.figure_page("nguyen", "5", tmp_path),
        "![Replication Fig 5](images/5/spikes_energy_v1.png)\n\n"
        "<!-- caption-5:start -->\nold\n<!-- caption-5:end -->\n\n**Status:** Open.\n\n"
        "<!-- gates-5:start -->\nold gates\n<!-- gates-5:end -->\n",
    )
    png = doc_paths.figure_images_dir("nguyen", "5", tmp_path) / "spikes_energy_v2.png"
    out = promote.promote_nguyen_5(manifest={"caption": "c", "png_version": 2}, png_path=png)
    assert out["png_link"] == "images/5/spikes_energy_v2.png"
    page_text = page.read_text()
    assert "![Replication Fig 5](images/5/spikes_energy_v2.png)" in page_text
    assert "old gates" not in page_text
    assert "**Status:** Open — see manifest gates (`spikes_energy_v2.png`)." in page_text
    tracker_text = tracker.read_text()
    assert "| Panel | Description | Status |" in tracker_text and "\nold\n" not in tracker_text
