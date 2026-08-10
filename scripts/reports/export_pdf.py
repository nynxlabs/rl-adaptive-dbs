#!/usr/bin/env python3
"""Export markdown outreach reports (vault ``reports/*.md``) to PDF.

Pipeline: preprocess Markdown (math, optional page breaks) → HTML → WeasyPrint.

Usage::

    uv sync --group reports
    uv run python -m rl_adaptive_dbs.run scripts/reports/export_pdf.py \\
        ~/knowledge-base/bme/brain-stimulation/rl-adaptive-dbs/reports/3.md \\
        -o ~/knowledge-base/bme/brain-stimulation/rl-adaptive-dbs/reports/3.pdf

    # One long scroll (no forced page breaks):
    uv run python -m rl_adaptive_dbs.run scripts/reports/export_pdf.py reports/3.md --continuous

Requires system libraries for WeasyPrint on Linux (Pango, Cairo). On Debian/Ubuntu::

    sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libcairo2
"""
from __future__ import annotations

import argparse
import re
import sys
from html import escape
from pathlib import Path

_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
_DISPLAY_MATH_RE = re.compile(r"\$\$([^$]+?)\$\$", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_BREAK_RE = re.compile(
    r"<div\s+style\s*=\s*[\"']page-break-after\s*:\s*always\s*;?\s*[\"']\s*>\s*</div>",
    re.IGNORECASE,
)
_PAGE_RULES_RE = re.compile(r"^@page[^{]*\{[^}]*\}\s*", re.MULTILINE)
_CONTINUOUS_PROBE_HEIGHT_PX = 30000
_CONTINUOUS_END_MARKER = '<div id="document-end"></div>'
# Matches ``report_pdf.css`` vertical ``@page`` margin (0.75in at 96 CSS px/in).
_PAGE_MARGIN_VERTICAL_PX = 0.75 * 96


def _latex_to_mathml(latex: str) -> str | None:
    try:
        from latex2mathml.converter import convert as latex2mathml_convert
    except ImportError:
        return None
    try:
        return latex2mathml_convert(latex.strip())
    except Exception:
        return None


def _replace_math(match: re.Match[str], *, display: bool) -> str:
    raw = match.group(1).strip()
    mathml = _latex_to_mathml(raw)
    if mathml is None:
        return f'<span class="math-fallback">{escape(raw)}</span>'
    wrapper = "div" if display else "span"
    klass = "math-display" if display else "math-inline"
    return f'<{wrapper} class="{klass}">{mathml}</{wrapper}>'


def preprocess_markdown(
    text: str,
    *,
    continuous: bool,
    strip_html_comments: bool,
) -> str:
    if strip_html_comments:
        text = _HTML_COMMENT_RE.sub("", text)
    if continuous:
        text = _PAGE_BREAK_RE.sub("", text)
    else:
        text = _PAGE_BREAK_RE.sub('<div class="page-break"></div>', text)
    text = _DISPLAY_MATH_RE.sub(lambda m: _replace_math(m, display=True), text)
    text = _INLINE_MATH_RE.sub(lambda m: _replace_math(m, display=False), text)
    return text


def markdown_to_html_body(text: str) -> str:
    import markdown

    return markdown.markdown(
        text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )


def build_html_document(
    body_html: str,
    *,
    continuous: bool,
    css_text: str,
    title: str,
) -> str:
    mode = "continuous" if continuous else "paginated"
    safe_title = escape(title)
    return (
        f"<!DOCTYPE html>\n"
        f"<html lang=\"en\" class=\"{mode}\">\n"
        "<head>\n"
        f"<meta charset=\"utf-8\">\n"
        f"<title>{safe_title}</title>\n"
        f"<style>\n{css_text}\n</style>\n"
        "</head>\n"
        f"<body class=\"{mode}\">\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )


def strip_page_rules(css_text: str) -> str:
    """Remove ``@page`` blocks so Python can inject exact page dimensions."""
    text = _PAGE_RULES_RE.sub("", css_text)
    text = re.sub(
        r"html\.(?:paginated|continuous)\s+body\s*\{[^}]*\}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text


def _page_rule(width: str, height_px: float) -> str:
    margin_in = _PAGE_MARGIN_VERTICAL_PX / 96
    return (
        f"@page {{ size: {width} {height_px}px; "
        f"margin: {margin_in}in 0.85in; }}"
    )


def _continuous_target_height(document) -> float:
    """Page height in px: content bottom (``#document-end``) plus bottom margin."""
    anchor = document.pages[0].anchors.get("document-end")
    if anchor is not None:
        end_y = max(anchor[1], anchor[3])
        return end_y + _PAGE_MARGIN_VERTICAL_PX
    page_box = document.pages[0]._page_box
    return float(page_box.height)


def write_continuous_pdf(
    html_doc: "HTML",
    output_pdf: Path,
    css_text: str,
) -> None:
    """Single-page PDF: probe render on a tall canvas, then exact height."""
    from weasyprint import CSS

    probe_css = CSS(
        string=css_text + _page_rule("8.5in", _CONTINUOUS_PROBE_HEIGHT_PX),
    )
    probe = html_doc.render(stylesheets=[probe_css])
    if len(probe.pages) != 1:
        raise SystemExit(
            f"continuous probe produced {len(probe.pages)} pages; "
            "content may exceed probe height — increase _CONTINUOUS_PROBE_HEIGHT_PX"
        )
    target_height = _continuous_target_height(probe)
    final_css = CSS(string=css_text + _page_rule("8.5in", target_height))
    final = html_doc.render(stylesheets=[final_css])
    if len(final.pages) > 1:
        # Rare rounding/layout drift: grow by one bottom margin and retry once.
        target_height += _PAGE_MARGIN_VERTICAL_PX
        final_css = CSS(string=css_text + _page_rule("8.5in", target_height))
        final = html_doc.render(stylesheets=[final_css])
    if len(final.pages) != 1:
        raise SystemExit(
            f"continuous final render produced {len(final.pages)} pages at "
            f"height {target_height}px; content may need a larger probe canvas"
        )
    html_doc.write_pdf(str(output_pdf), stylesheets=[final_css])


def export_pdf(
    input_md: Path,
    output_pdf: Path,
    *,
    continuous: bool,
    strip_html_comments: bool,
    css_path: Path,
) -> None:
    try:
        from weasyprint import HTML
    except ImportError as exc:
        msg = (
            "WeasyPrint is not installed. Run: uv sync --group reports\n"
            "On Linux you may also need: sudo apt install libpango-1.0-0 "
            "libpangocairo-1.0-0 libcairo2"
        )
        raise SystemExit(msg) from exc

    input_md = input_md.resolve()
    if not input_md.is_file():
        raise SystemExit(f"input not found: {input_md}")

    css_text = css_path.read_text(encoding="utf-8")
    if continuous:
        css_text = strip_page_rules(css_text)
    source = input_md.read_text(encoding="utf-8")
    processed = preprocess_markdown(
        source,
        continuous=continuous,
        strip_html_comments=strip_html_comments,
    )
    body_html = markdown_to_html_body(processed)
    if continuous:
        body_html += _CONTINUOUS_END_MARKER
    title = input_md.stem
    document = build_html_document(
        body_html,
        continuous=continuous,
        css_text=css_text,
        title=title,
    )

    output_pdf = output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    base_url = str(input_md.parent)
    html_doc = HTML(string=document, base_url=base_url)
    if continuous:
        write_continuous_pdf(html_doc, output_pdf, css_text)
    else:
        from weasyprint import CSS

        html_doc.write_pdf(str(output_pdf), stylesheets=[CSS(string=css_text)])
    print(f"wrote {output_pdf}", flush=True)


def default_output_path(input_md: Path) -> Path:
    return input_md.with_suffix(".pdf")


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_md",
        type=Path,
        help="Markdown report (e.g. vault reports/3.md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: same name as input, .pdf)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Single flowing page: drop forced page-break divs and use @page size auto",
    )
    parser.add_argument(
        "--keep-html-comments",
        action="store_true",
        help="Keep <!-- report3-* marker comments --> in the PDF source",
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=here / "report_pdf.css",
        help="CSS file for print layout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output or default_output_path(args.input_md)
    export_pdf(
        args.input_md,
        output,
        continuous=args.continuous,
        strip_html_comments=not args.keep_html_comments,
        css_path=args.css,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
