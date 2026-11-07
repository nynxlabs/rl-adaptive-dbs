"""LICENSES.md is current with licenses.yaml, and no piece breaks the vendor/link rules."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("licenses_tool", ROOT / "tools" / "licenses.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_map_follows_the_rules():
    assert tool.problems(tool.load()) == []


def test_table_is_current():
    text = tool.DOC.read_text()
    assert tool.render(text, tool.load()) == text, "run: uv run python tools/licenses.py --write"


def test_rules_catch_breaks():
    bad = [
        {"id": "a", "name": "a", "kind": "code", "mode": "vendored", "license": "GPL-3.0"},
        {"id": "b", "name": "b", "kind": "code", "mode": "linked", "license": "none"},
        {"id": "c", "name": "c", "kind": "code", "mode": "excluded", "license": "none"},
    ]
    got = tool.problems(bad)
    assert any(p.startswith("a:") and "never vendored" in p for p in got)
    assert any(p.startswith("b:") and "stays out" in p for p in got)
    assert any(p.startswith("c:") and "until" in p for p in got)
