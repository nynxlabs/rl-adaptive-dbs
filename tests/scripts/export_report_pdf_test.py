"""Tests for scripts/reports/export_pdf.py (no WeasyPrint required)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from reports.export_pdf import (  # noqa: E402
    build_html_document,
    markdown_to_html_body,
    preprocess_markdown,
    strip_page_rules,
)


def test_preprocess_strips_page_break_when_continuous() -> None:
    src = "hello\n<div style=\"page-break-after: always;\"></div>\nworld"
    out = preprocess_markdown(src, continuous=True, strip_html_comments=True)
    assert "page-break" not in out
    assert "hello" in out and "world" in out


def test_preprocess_keeps_page_break_class_when_paginated() -> None:
    src = "<div style=\"page-break-after: always;\"></div>"
    out = preprocess_markdown(src, continuous=False, strip_html_comments=False)
    assert "page-break" in out


def test_inline_math_replaced() -> None:
    src = "beta $P_\\beta$ power"
    out = preprocess_markdown(src, continuous=True, strip_html_comments=True)
    assert "<math" in out or "math-fallback" in out
    assert "$P_" not in out


def test_markdown_table_and_image() -> None:
    body = markdown_to_html_body("| A | B |\n|---|---|\n| 1 | 2 |")
    assert "<table>" in body


def test_strip_page_rules() -> None:
    css = "@page { size: letter; }\nbody { color: red; }\n@page continuous { size: auto; }"
    out = strip_page_rules(css)
    assert "@page" not in out
    assert "color: red" in out


def test_strip_page_rules_ignores_comment_atpage() -> None:
    css = "/* note about page rules */\n@page { size: letter; }\nbody { margin: 0; }"
    out = strip_page_rules(css)
    assert "note about page rules" in out
    assert "@page" not in out


def test_build_html_document_modes() -> None:
    html = build_html_document("<p>hi</p>", continuous=True, css_text="body{}", title="t")
    assert "class=\"continuous\"" in html
