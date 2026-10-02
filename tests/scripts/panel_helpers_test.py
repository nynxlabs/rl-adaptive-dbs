"""rl_adaptive_dbs.panel: gate verdict normalization, exit codes, script loading, provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_adaptive_dbs import panel


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"gates": {"pass": True}}, True),
        ({"gates": {"all_pass": False}}, False),
        ({"gates_pass": True}, True),
        ({"summary": {"gates_pass": False, "automation_pass": True}}, False),
        ({"panel": {"gates": {"all_pass": True}}}, True),
        ({"panel": {"gates_pass": False}}, False),
        ({"caption": "no gates"}, None),
        (None, None),
    ],
)
def test_gates_pass_layouts(manifest: dict | None, expected: bool | None) -> None:
    assert panel.gates_pass(manifest) is expected


def test_exit_codes() -> None:
    assert panel.exit_code({"gates": {"pass": True}}) == panel.EXIT_PASS
    assert panel.exit_code({"gates": {"pass": False}}) == panel.EXIT_GATE_FAIL
    assert panel.exit_code({}) == panel.EXIT_GATE_FAIL
    assert panel.exit_code({"gates": {"pass": False}}, smoke=True) == panel.EXIT_PASS


def test_load_script_module_supports_dataclasses(tmp_path: Path) -> None:
    script = tmp_path / "helper.py"
    script.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\nclass Point:\n    x: int\n",
        encoding="utf-8",
    )
    module = panel.load_script_module("panel_test_helper_module", script)
    assert module.Point(3).x == 3


def test_stamp_manifest_records_provenance(tmp_path: Path) -> None:
    data = tmp_path / "series.json"
    data.write_text("{}", encoding="utf-8")
    manifest = panel.stamp_manifest({"gates": {"pass": False}}, argv=["--x"], inputs={"series": data})
    prov = manifest["provenance"]
    assert manifest["gates_pass"] is False
    assert prov["argv"] == ["--x"]
    assert len(prov["inputs"]["series"]["sha256"]) == 64
    assert "commit" in prov["git"]
