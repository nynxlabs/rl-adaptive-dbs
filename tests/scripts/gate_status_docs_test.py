"""Gate-status scripts write each panel's gate table in its figure page (tracker fallback)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs.panel import load_script_module

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "digitization"))

_RGS = load_script_module(
    "ravivarapu_gate_status_docs_test", REPO / "scripts/digitization/ravivarapu_gate_status.py"
)


def _block(marker: str) -> str:
    return f"<!-- {marker}:start -->\nold\n<!-- {marker}:end -->\n"


def _fake_status(panel: str):
    return _RGS.PanelGateStatus(
        panel=panel,
        pass_field="pass",
        overall=True,
        shape_overall=True,
        gates={"g": True},
        source=f"fake/{panel}.json",
        rows=(_RGS.GateRow("g", "fake gate"),),
    )


def _tree(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    tracker = doc_paths.tracker_path("ravivarapu", tmp_path)
    tracker.parent.mkdir(parents=True)
    # 4a has not moved yet: its gate block is still in the tracker.
    tracker.write_text(_block("summary") + _block("gates-4a"), encoding="utf-8")
    pages = {}
    for panel in _RGS.RAVIVARAPU_GATE_ROWS:
        if panel == "4a":
            continue
        page_name = doc_paths.panel_for_marker_id("ravivarapu", panel)
        page = doc_paths.figure_page("ravivarapu", page_name, tmp_path)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"# {panel}\n" + _block(f"gates-{panel}"), encoding="utf-8")
        pages[panel] = page
    return tracker, pages


def test_gate_tables_land_in_pages_and_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_RGS, "evaluate_panel", _fake_status)
    tracker, pages = _tree(tmp_path)
    _RGS.refresh_gate_tables(tracker)
    assert pages["t2"].name == "table2.md"
    for page in pages.values():
        assert "fake gate" in page.read_text() and "\nold\n" not in page.read_text()
    text = tracker.read_text()
    assert "fake gate" in text  # gates-4a fallback
    assert "| Panel | Description | Status |" in text  # summary


def test_missing_gate_block_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_RGS, "evaluate_panel", _fake_status)
    tracker, pages = _tree(tmp_path)
    pages["6"].write_text("# 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing gates markers for panel 6"):
        _RGS.refresh_gate_tables(tracker)
