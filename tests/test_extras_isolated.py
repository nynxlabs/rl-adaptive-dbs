"""The Apache-2.0 core never imports an opt-in extra (extras/ holds differently licensed glue)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ("plants", "controllers", "adapters")
PATTERN = re.compile(r"^\s*(from|import)\s+(extras|fleming2020)\b", re.M)


def test_core_never_imports_extras():
    hits = [str(p.relative_to(ROOT)) for pkg in CORE for p in (ROOT / pkg).rglob("*.py")
            if PATTERN.search(p.read_text())]
    assert hits == []
