"""Render LICENSES.md's table from licenses.yaml, and check the map against the vendor/link rules.

    uv run python tools/licenses.py            # check (exit 1 on a rule break or a stale table)
    uv run python tools/licenses.py --write    # rewrite the table in LICENSES.md

Paper licenses and AI rules are read from each component's pawtrace spec (`spec:`), never
repeated in licenses.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "licenses.yaml"
DOC = ROOT / "LICENSES.md"
START, END = "<!-- licenses-table:start -->", "<!-- licenses-table:end -->"
MODES = ("own", "vendored", "linked", "reference", "excluded")
VENDORABLE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "CC-BY-4.0", "CC0-1.0", "public-domain"}
COPYLEFT = ("GPL", "LGPL", "AGPL")


def load() -> list[dict]:
    return yaml.safe_load(MAP.read_text())["components"]


def spec_rights(c: dict) -> dict | None:
    if not c.get("spec"):
        return None
    p = ROOT / c["spec"] / "spec.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get("rights") if p.exists() else None


def problems(components: list[dict]) -> list[str]:
    out, seen = [], set()
    for c in components:
        cid, mode, lic = c.get("id"), c.get("mode"), str(c.get("license", ""))
        if cid in seen:
            out.append(f"{cid}: duplicate id")
        seen.add(cid)
        if mode not in MODES:
            out.append(f"{cid}: mode {mode!r} not one of {MODES}")
        if mode == "own" and lic != "Apache-2.0":
            out.append(f"{cid}: our own code is Apache-2.0")
        if mode == "vendored":
            if lic not in VENDORABLE:
                out.append(f"{cid}: {lic} may not be vendored (link it, or leave it out)")
            if not c.get("license_file") or not (ROOT / c["license_file"]).exists():
                out.append(f"{cid}: vendored code needs its license_file kept next to it")
        if lic == "none" and mode != "excluded" and c.get("kind") == "code":
            out.append(f"{cid}: code with no license stays out entirely (mode: excluded)")
        if any(lic.startswith(k) for k in COPYLEFT) and mode in ("vendored", "own"):
            out.append(f"{cid}: {lic} code is never vendored; link it as an opt-in module")
        if "NC" in lic and mode == "vendored":
            out.append(f"{cid}: non-commercial terms are never vendored")
        if mode == "excluded" and not c.get("until"):
            out.append(f"{cid}: excluded needs `until` (what would let it in)")
        if c.get("spec") and c.get("note", "").find("lands with") < 0 and spec_rights(c) is None:
            out.append(f"{cid}: spec {c['spec']} not found")
    return out


def table(components: list[dict]) -> str:
    rows = ["| Piece | Kind | Mode | License | AI tools | Credit / source |", "|---|---|---|---|---|---|"]
    for c in components:
        r = spec_rights(c)
        if c.get("kind") == "paper" or (r and c.get("mode") == "reference"):
            lic = (r or {}).get("license", "see spec") if r else "see spec"
        else:
            lic = c.get("license", "")
        if r and c.get("kind") != "paper" and c.get("mode") == "own":
            lic = f"{c.get('license')} (paper: {r.get('license')})"
        ai = (r or {}).get("rules", {}).get("ai_tools", "") if r else ""
        credit = c.get("credit", "")
        src = c.get("source") or c.get("path") or c.get("spec") or ""
        extra = "; ".join(x for x in (c.get("until") and f"until: {c['until']}", c.get("note")) if x)
        cell = " · ".join(x for x in (credit, f"`{src}`" if src else "", extra) if x)
        rows.append(f"| {c['name']} | {c.get('kind','')} | {c.get('mode','')} | {lic} | {ai} | {cell} |")
    return "\n".join(rows)


def render(text: str, components: list[dict]) -> str:
    a, b = text.index(START) + len(START), text.index(END)
    return text[:a] + "\n" + table(components) + "\n" + text[b:]


def main(argv: list[str]) -> int:
    comps = load()
    bad = problems(comps)
    for p in bad:
        print("error ", p)
    text = DOC.read_text()
    new = render(text, comps)
    if "--write" in argv:
        DOC.write_text(new)
        print("wrote LICENSES.md")
    elif new != text:
        print("error  LICENSES.md table is stale: run `uv run python tools/licenses.py --write`")
        bad.append("stale")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
