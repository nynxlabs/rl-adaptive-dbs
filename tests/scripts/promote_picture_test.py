"""The light/dark ``<picture>`` markup promote writes into the figure docs."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from figures.papers.overlay_import import load_paper_overlay  # noqa: E402
from rl_adaptive_dbs.doc_paths import DocSet  # noqa: E402

_overlay = load_paper_overlay()


def _load_promote():
    import importlib.util

    path = _ROOT / "scripts" / "figures" / "papers" / "promote.py"
    spec = importlib.util.spec_from_file_location("figure_promote_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


promote = _load_promote()

PICTURE = (
    '<picture><source media="(prefers-color-scheme: dark)" srcset="images/4a/training_beta_v48_dark.png">'
    '<img src="images/4a/training_beta_v48.png" alt="Replication Fig 4a"></picture>'
)


def test_dark_suffix_matches_figure_style() -> None:
    assert promote.DARK_SUFFIX == _overlay.figure_style.DARK_SUFFIX
    assert promote.dark_rel("docs/x/images/4a/a_v3.png") == "docs/x/images/4a/a_v3_dark.png"


def test_picture_markup_exact_form() -> None:
    got = promote.picture_markup(
        "Replication Fig 4a", "images/4a/training_beta_v48.png", "images/4a/training_beta_v48_dark.png"
    )
    assert got == PICTURE
    with_width = promote.picture_markup("A", "a.png", "a_dark.png", attrs=' width="280"')
    assert with_width.endswith('<img src="a.png" alt="A" width="280"></picture>')
    assert promote.picture_markup("A", "a.png", None) == "![A](a.png)"


def _docset(tmp_path: Path, text: str, monkeypatch, *, dark: bool = True) -> tuple[DocSet, Path]:
    root = tmp_path
    doc = root / "docs" / "controllers" / "ddpg" / "figures" / "4a.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(text, encoding="utf-8")
    images = doc.parent / "images" / "4a"
    images.mkdir(parents=True)
    for name in ("training_beta_v49.png", *(("training_beta_v49_dark.png",) if dark else ())):
        (images / name).write_bytes(b"png")
    monkeypatch.setattr(promote, "CHECKOUT_ROOT", root)
    monkeypatch.setattr(promote, "REPO_ROOT", root)
    return DocSet([doc], root=root), doc


REL = "docs/controllers/ddpg/figures/images/4a/training_beta_v49.png"
NEW = PICTURE.replace("v48", "v49")


def test_set_image_link_converts_markdown_to_picture(tmp_path: Path, monkeypatch) -> None:
    docs, doc = _docset(tmp_path, "a\n\n![Replication Fig 4a](images/4a/training_beta_v48.png)\n\nb\n", monkeypatch)
    promote.set_image_link(docs, alt="Replication Fig 4a", repo_rel=REL)
    docs.save()
    assert doc.read_text() == f"a\n\n{NEW}\n\nb\n"


def test_set_image_link_updates_picture_and_keeps_img_attrs(tmp_path: Path, monkeypatch) -> None:
    old = PICTURE.replace('alt="Replication Fig 4a">', 'alt="Replication Fig 4a" width="280">')
    docs, doc = _docset(tmp_path, f"| {old} |\n", monkeypatch)
    promote.set_image_link(docs, alt="Replication Fig 4a", repo_rel=REL)
    docs.save()
    assert doc.read_text() == "| " + NEW.replace('alt="Replication Fig 4a">', 'alt="Replication Fig 4a" width="280">') + " |\n"


def test_set_image_link_without_dark_variant_stays_markdown(tmp_path: Path, monkeypatch) -> None:
    docs, doc = _docset(tmp_path, "![Replication Fig 4a](images/4a/training_beta_v48.png)\n", monkeypatch, dark=False)
    promote.set_image_link(docs, alt="Replication Fig 4a", repo_rel=REL)
    docs.save()
    assert doc.read_text() == "![Replication Fig 4a](images/4a/training_beta_v49.png)\n"
