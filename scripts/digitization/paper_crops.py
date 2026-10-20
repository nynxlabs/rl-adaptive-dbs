"""Build the paper figure crops (``paper.png``) from your own copy of each paper.

The digitization tools (``extract_pil.py``, ``pil_to_wpd.py``,
``nguyen_fig3_boxplot.py``) and the configs in ``figs/`` read pixel boxes off
``docs/controllers/<controller>/figures/images/<panel>/paper.png``. Those crops are the publishers'
figures, so they are **not** in this repository: each user builds them locally
from their own copy of the papers. ``paper.png`` is gitignored.

``paper_crops.json`` (next to this file) is the committed spec: for each panel,
the PDF page, the crop box in PDF points (origin top-left, y down) and the
output size in pixels. A crop built from the spec has the same pixel geometry
as the crops the ``figs/`` configs were measured on.

Put your PDFs in ``reference-material/papers/`` (gitignored), named as the
spec says — ``mehregan.pdf``, ``nguyen.pdf``, ``ravivarapu.pdf`` — or point
``--papers-dir`` / the ``RL_DBS_PAPERS_DIR`` environment variable at another
folder that holds them under those names. Then::

    uv run --group papers python scripts/digitization/paper_crops.py build
    uv run --group papers python scripts/digitization/paper_crops.py build --panel mehregan/4a

``build --arxiv`` instead builds the Ravivarapu panels from the arXiv version
of that paper (arXiv:2507.06326v1, CC BY 4.0) into ``arxiv_v1.png``, which is
committed. It downloads the arXiv PDF into the papers folder if it is missing.

Other subcommands:

* ``verify`` builds every panel in memory and compares it with existing crops
  (``--reference-root``, default the repo) — size, correlation, mean abs diff.
* ``calibrate`` finds the page, crop box and scale of each panel by matching
  a reference crop against the rendered PDF pages (normalized
  cross-correlation via FFT over a scale search), and with ``--write`` stores
  the result in the spec. This is how the spec was made; you only need it to
  add a panel or a new source.

Rendering uses pypdfium2 (``--group papers``, wheels for Windows, macOS and
Linux) when it is installed, else Poppler's ``pdftoppm`` if it is on PATH.
The spec was calibrated with pypdfium2, which reproduces the original crops
pixel for pixel; ``pdftoppm`` renders the same box with its own anti-aliasing
(grayscale correlation about 0.9 with the pypdfium2 crop), so prefer
pypdfium2 before measuring pixels. Pillow is needed in either case (also in
``--group papers``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - reported at run time
    Image = None

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("paper_crops.json")
DEFAULT_PAPERS_DIR = REPO / "reference-material" / "papers"
PAPERS_DIR_ENV = "RL_DBS_PAPERS_DIR"

# Calibration search: coarse page/scale scan, then two fine passes.
COARSE_SCALE = 1.5          # px per pt for the coarse page renders (108 dpi)
SCALE_RANGE = (1.0, 5.0)    # px per pt searched (72-360 dpi)
COARSE_STEP = 0.02          # relative scale step of the coarse scan
FINE_PASSES = ((0.002, 15), (0.0002, 10))  # (relative step, half-width in steps)


# ---------------------------------------------------------------- spec / paths


def load_spec(path: Path = SPEC_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_spec(spec: dict, path: Path = SPEC_PATH) -> None:
    text = json.dumps(spec, indent=2)
    # Keep short lists (boxes, sizes, aliases) on one line.
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)
    path.write_text(text + "\n", encoding="utf-8")


def _notes_papers_dir() -> Path | None:
    """``<notes_dir>/papers`` when a notes folder is configured (``RL_DBS_NOTES_DIR`` /
    ``export.notes_dir``) or detected from symlinked ``replications.md`` trackers."""
    try:
        from rl_adaptive_dbs.notes_export import resolve_notes_dir
    except Exception:  # package not importable (e.g. run outside the project venv)
        return None
    trackers = sorted((REPO / "figures").glob("*/replications.md"))
    try:
        notes = resolve_notes_dir(trackers, repo_root=REPO)
    except Exception:
        return None
    return notes / "papers" if notes is not None else None


def papers_dirs(cli_value: str | None) -> list[Path]:
    """Folders searched for the PDFs, in order: ``--papers-dir``, ``RL_DBS_PAPERS_DIR``,
    ``<notes_dir>/papers/``, ``reference-material/papers/``."""
    dirs: list[Path] = []
    if cli_value:
        dirs.append(Path(cli_value).expanduser())
    env = os.environ.get(PAPERS_DIR_ENV)
    if env:
        dirs.append(Path(env).expanduser())
    notes = _notes_papers_dir()
    if notes is not None:
        dirs.append(notes)
    dirs.append(DEFAULT_PAPERS_DIR)
    out: list[Path] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def panel_paper(panel: str) -> str:
    return panel.split("/", 1)[0]


def output_path(root: Path, panel: str, name: str) -> Path:
    paper, sub = panel.split("/", 1)
    return root / "figures" / paper / "images" / sub / name


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CropError(RuntimeError):
    pass


def find_pdf(source: dict, paper: str, pdirs: list[Path], *, download: bool = False) -> Path:
    """First ``file`` or ``aliases`` name of ``paper`` found in ``pdirs``."""
    meta = source["papers"].get(paper)
    if meta is None:
        raise CropError(f"no paper {paper!r} in this source of the spec")
    names = [meta["file"], *meta.get("aliases", [])]
    pdf = next((d / n for d in pdirs for n in names if (d / n).is_file()), None)
    if pdf is None and download and meta.get("url"):
        # Explicit --papers-dir / RL_DBS_PAPERS_DIR, else reference-material/papers/.
        target = pdirs[0] if pdirs[0] != _notes_papers_dir() else DEFAULT_PAPERS_DIR
        target.mkdir(parents=True, exist_ok=True)
        pdf = target / meta["file"]
        print(f"downloading {meta['url']} -> {pdf}")
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "rl-adaptive-dbs paper_crops"})
        with urllib.request.urlopen(req, timeout=120) as resp, pdf.open("wb") as out:
            shutil.copyfileobj(resp, out)
    if pdf is None:
        raise CropError(
            f"no PDF for {paper}: expected your copy of {meta.get('citation', paper)}"
            + (f" (doi:{meta['doi']})" if meta.get("doi") else "")
            + f"\n  named {' or '.join(repr(n) for n in names)} in one of: "
            + ", ".join(str(d) for d in pdirs)
            + f"\n  (pass --papers-dir or set {PAPERS_DIR_ENV} to use another folder)"
        )
    want = meta.get("sha256")
    if want:
        got = sha256_of(pdf)
        if got != want:
            print(
                f"warning: {pdf.name} sha256 {got[:12]}... differs from the spec's {want[:12]}...;"
                " a different download of the same paper can differ (e.g. a download stamp)."
                " Building anyway; run `verify` if the crops look off.",
                file=sys.stderr,
            )
    return pdf


# ------------------------------------------------------------------ rendering


def _renderer() -> str:
    try:
        import pypdfium2  # noqa: F401

        return "pdfium"
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        return "pdftoppm"
    raise CropError(
        "no PDF renderer: install the papers group (uv sync --group papers) for pypdfium2,"
        " or put Poppler's pdftoppm on PATH"
    )


class Renderer:
    """Render pixel regions of PDF pages at a scale in px per pt.

    The pixel grid is the full-page grid at that scale: pixel ``(X, Y)``
    covers PDF points ``[X/s, (X+1)/s)`` from the top-left corner.
    """

    def __init__(self, kind: str | None = None):
        if Image is None:
            raise CropError("Pillow is missing: uv sync --group papers")
        self.kind = kind or _renderer()
        if self.kind == "pdftoppm":
            print(
                "note: rendering with pdftoppm; its anti-aliasing differs from pypdfium2, which the"
                " spec was calibrated with (same geometry, slightly different pixels)."
                " For exact crops: uv sync --group papers",
                file=sys.stderr,
            )
        self._docs: dict[Path, object] = {}
        self._pages: dict[tuple, np.ndarray] = {}  # last full-page pdfium render

    def _doc(self, pdf: Path):
        import pypdfium2 as pdfium

        if pdf not in self._docs:
            self._docs[pdf] = pdfium.PdfDocument(str(pdf))
        return self._docs[pdf]

    def page_count(self, pdf: Path) -> int:
        if self.kind == "pdfium":
            return len(self._doc(pdf))
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
        raise CropError(f"could not read the page count of {pdf}")

    def page_size_pt(self, pdf: Path, page: int) -> tuple[float, float]:
        if self.kind == "pdfium":
            w, h = self._doc(pdf)[page - 1].get_size()
            return float(w), float(h)
        out = subprocess.run(
            ["pdfinfo", "-f", str(page), "-l", str(page), str(pdf)], capture_output=True, text=True, check=True
        ).stdout
        for line in out.splitlines():
            if line.startswith(f"Page") and "size:" in line:
                parts = line.split("size:")[1].split()
                return float(parts[0]), float(parts[2])
        raise CropError(f"could not read the page size of {pdf} page {page}")

    def region(self, pdf: Path, page: int, scale: float, x: int, y: int, w: int, h: int) -> np.ndarray:
        """RGB uint8 array (h, w, 3) of the pixel box at ``scale``; white outside the page."""
        pw, ph = self.page_size_pt(pdf, page)
        full_w, full_h = int(np.ceil(pw * scale)), int(np.ceil(ph * scale))
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, full_w), min(y + h, full_h)
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        if x1 <= x0 or y1 <= y0:
            return out
        if self.kind == "pdfium":
            # Render the whole page and slice it, so a crop never depends on the
            # clip pdfium would otherwise resample the page's images within.
            key = (pdf, page, scale)
            if key not in self._pages:
                self._pages = {key: np.asarray(self._doc(pdf)[page - 1].render(scale=scale).to_pil().convert("RGB"))}
            img = self._pages[key][y0:y1, x0:x1]
        else:
            with tempfile.TemporaryDirectory() as td:
                prefix = Path(td) / "page"
                subprocess.run(
                    [
                        # Same page-to-pixel mapping as pdfium: the page spans
                        # ceil(size * scale) pixels on each axis.
                        "pdftoppm", "-f", str(page), "-l", str(page),
                        "-scale-to-x", str(full_w), "-scale-to-y", str(full_h),
                        "-x", str(x0), "-y", str(y0), "-W", str(x1 - x0), "-H", str(y1 - y0),
                        "-png", "-singlefile", str(pdf), str(prefix),
                    ],
                    check=True,
                )
                with Image.open(prefix.with_suffix(".png")) as im:
                    img = im.convert("RGB")
        arr = np.asarray(img)
        rh, rw = min(arr.shape[0], y1 - y0), min(arr.shape[1], x1 - x0)
        out[y0 - y : y0 - y + rh, x0 - x : x0 - x + rw] = arr[:rh, :rw]
        return out

    def page(self, pdf: Path, page: int, scale: float) -> np.ndarray:
        pw, ph = self.page_size_pt(pdf, page)
        return self.region(pdf, page, scale, 0, 0, int(np.ceil(pw * scale)), int(np.ceil(ph * scale)))


def panel_geometry(entry: dict) -> tuple[float, int, int, int, int]:
    """(scale, X, Y, W, H) in pixels for a spec entry."""
    x0, y0, x1, y1 = entry["box_pt"]
    w, h = entry["size_px"]
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    if abs(sx - sy) / sx > 1e-3:
        raise CropError(f"box_pt and size_px disagree on the scale ({sx:.4f} vs {sy:.4f} px/pt)")
    scale = (sx + sy) / 2
    return scale, int(round(x0 * scale)), int(round(y0 * scale)), int(w), int(h)


def render_panel(renderer: Renderer, pdf: Path, entry: dict) -> np.ndarray:
    scale, x, y, w, h = panel_geometry(entry)
    return renderer.region(pdf, int(entry["page"]), scale, x, y, w, h)


# ------------------------------------------------------------------- matching


def to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB"))


def _window_sum(a: np.ndarray, h: int, w: int) -> np.ndarray:
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


def ncc_map(image: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of ``tmpl`` at every valid offset in ``image``."""
    from scipy.signal import fftconvolve

    h, w = tmpl.shape
    if image.shape[0] < h or image.shape[1] < w:
        return np.full((1, 1), -1.0)
    t = tmpl - tmpl.mean()
    tn = np.sqrt((t * t).sum())
    if tn == 0:
        return np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1))
    num = fftconvolve(image, t[::-1, ::-1], mode="valid")
    n = h * w
    s1 = _window_sum(image, h, w)
    s2 = _window_sum(image * image, h, w)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = np.sqrt(var) * tn
    return np.where(den > 1e-6 * tn * 255, num / np.maximum(den, 1e-12), 0.0)


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(Pearson correlation of grayscale, mean abs diff over RGB in 0-255)."""
    ga, gb = to_gray(a).ravel(), to_gray(b).ravel()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    den = np.sqrt((ga * ga).sum() * (gb * gb).sum())
    corr = float((ga * gb).sum() / den) if den else float("nan")
    mad = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())
    return corr, mad


def _resize_gray(g: np.ndarray, factor: float) -> np.ndarray:
    h, w = g.shape
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
    return np.asarray(im.resize(size, Image.LANCZOS), dtype=np.float64)


@dataclass
class Match:
    page: int
    scale: float
    x: int
    y: int
    score: float

    def entry(self, w: int, h: int) -> dict:
        s = self.scale
        return {
            "page": self.page,
            "box_pt": [round(self.x / s, 4), round(self.y / s, 4), round((self.x + w) / s, 4), round((self.y + h) / s, 4)],
            "size_px": [w, h],
        }


def calibrate_one(
    renderer: Renderer,
    pdf: Path,
    ref_rgb: np.ndarray,
    pages: list[int] | None = None,
    start: Match | None = None,
) -> Match:
    """Locate ``ref_rgb`` in ``pdf``: coarse page/scale scan (skipped when ``start``
    is given), then fine scale passes rendered at full resolution."""
    tmpl = to_gray(ref_rgb)
    h, w = tmpl.shape
    best = start if start is not None else _coarse(renderer, pdf, tmpl, pages)
    return _refine(renderer, pdf, tmpl, best)


def _coarse(renderer: Renderer, pdf: Path, tmpl: np.ndarray, pages: list[int] | None) -> Match:
    pages = pages or list(range(1, renderer.page_count(pdf) + 1))
    scales = []
    s = SCALE_RANGE[0]
    while s <= SCALE_RANGE[1]:
        scales.append(s)
        s *= 1 + COARSE_STEP
    best: Match | None = None
    for page in pages:
        img = to_gray(renderer.page(pdf, page, COARSE_SCALE))
        for s in scales:
            t = _resize_gray(tmpl, COARSE_SCALE / s)
            if t.shape[0] < 8 or t.shape[1] < 8:
                continue
            m = ncc_map(img, t)
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            score = float(m[iy, ix])
            if best is None or score > best.score:
                best = Match(page, s, int(round(ix * s / COARSE_SCALE)), int(round(iy * s / COARSE_SCALE)), score)
    if best is None:
        raise CropError("no match found")
    return best


def _refine(renderer: Renderer, pdf: Path, tmpl: np.ndarray, best: Match) -> Match:
    h, w = tmpl.shape
    for rel, half in FINE_PASSES:
        centre, best = best, None
        for k in range(-half, half + 1):
            s = centre.scale * (1 + k * rel)
            # Search a window around the current estimate, rescaled to s.
            cx, cy = centre.x * s / centre.scale, centre.y * s / centre.scale
            margin = int(max(12, 0.04 * max(w, h) + abs(k) * rel * max(w, h) * 2))
            X, Y = int(round(cx)) - margin, int(round(cy)) - margin
            region = to_gray(renderer.region(pdf, centre.page, s, X, Y, w + 2 * margin, h + 2 * margin))
            m = ncc_map(region, tmpl)
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            score = float(m[iy, ix])
            if best is None or score > best.score:
                best = Match(centre.page, s, X + int(ix), Y + int(iy), score)
    return best


# ------------------------------------------------------------------ commands


def _select(source: dict, panels: list[str] | None) -> list[str]:
    names = list(source["panels"])
    if not panels:
        return names
    unknown = [p for p in panels if p not in source["panels"]]
    if unknown:
        raise CropError(f"not in the spec: {', '.join(unknown)} (known: {', '.join(names)})")
    return panels


def cmd_build(args: argparse.Namespace) -> int:
    spec = load_spec()
    source_name = "arxiv" if args.arxiv else "publisher"
    source = spec["sources"][source_name]
    pdir = papers_dirs(args.papers_dir)
    out_root = Path(args.out_root) if args.out_root else REPO
    renderer = Renderer(args.renderer)
    failed = 0
    for panel in _select(source, args.panel):
        entry = source["panels"][panel]
        try:
            pdf = find_pdf(source, panel_paper(panel), pdir, download=args.arxiv)
        except CropError as exc:
            print(f"error: {panel}: {exc}", file=sys.stderr)
            failed += 1
            continue
        arr = render_panel(renderer, pdf, entry)
        out = output_path(out_root, panel, source["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_symlink():
            print(f"{panel}: {out} is a symlink; left as is (remove it to rebuild)", file=sys.stderr)
            continue
        Image.fromarray(arr).save(out)
        print(f"{panel}: {out} ({arr.shape[1]}x{arr.shape[0]}, page {entry['page']}, {renderer.kind})")
    return 1 if failed else 0


def cmd_verify(args: argparse.Namespace) -> int:
    spec = load_spec()
    source_name = "arxiv" if args.arxiv else "publisher"
    source = spec["sources"][source_name]
    pdir = papers_dirs(args.papers_dir)
    ref_root = Path(args.reference_root) if args.reference_root else REPO
    ref_name = args.reference_name or source["output"]
    renderer = Renderer(args.renderer)
    bad = 0
    print(f"{'panel':16} {'size':>9} {'corr':>7} {'mad':>6}  ({renderer.kind}, reference {ref_name})")
    for panel in _select(source, args.panel):
        ref = output_path(ref_root, panel, ref_name)
        if not ref.exists():
            print(f"{panel:16} {'-':>9} {'-':>7} {'-':>6}  no reference {ref}")
            continue
        try:
            pdf = find_pdf(source, panel_paper(panel), pdir)
        except CropError as exc:
            print(f"error: {panel}: {exc}", file=sys.stderr)
            bad += 1
            continue
        built = render_panel(renderer, pdf, source["panels"][panel])
        want = load_rgb(ref)
        size = f"{built.shape[1]}x{built.shape[0]}"
        if built.shape != want.shape:
            print(f"{panel:16} {size:>9} {'-':>7} {'-':>6}  size differs from reference {want.shape[1]}x{want.shape[0]}")
            bad += 1
            continue
        corr, mad = similarity(built, want)
        flag = "" if corr >= args.min_corr else f"  below {args.min_corr}"
        bad += bool(flag)
        print(f"{panel:16} {size:>9} {corr:7.4f} {mad:6.2f}{flag}")
    return 1 if bad else 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    spec = load_spec()
    source_name = "arxiv" if args.arxiv else "publisher"
    source = spec["sources"][source_name]
    pdir = papers_dirs(args.papers_dir)
    ref_root = Path(args.reference_root) if args.reference_root else REPO
    ref_name = args.reference_name or "paper.png"
    renderer = Renderer(args.renderer)
    panels = args.panel or list(source["panels"])
    changed = False
    for panel in panels:
        ref = output_path(ref_root, panel, ref_name)
        if not ref.exists():
            print(f"{panel}: no reference crop at {ref}", file=sys.stderr)
            continue
        pdf = find_pdf(source, panel_paper(panel), pdir, download=args.arxiv)
        ref_rgb = load_rgb(ref)
        pages = [args.page] if args.page else None
        start = None
        if args.refine:
            old_entry = source["panels"].get(panel) or {}
            if "box_pt" not in old_entry:
                raise CropError(f"{panel}: --refine needs an existing box in the spec")
            sc, x, y, _, _ = panel_geometry(old_entry)
            start = Match(int(old_entry["page"]), sc, x, y, float("-inf"))
        m = calibrate_one(renderer, pdf, ref_rgb, pages, start)
        entry = m.entry(ref_rgb.shape[1], ref_rgb.shape[0])
        corr, mad = similarity(render_panel(renderer, pdf, entry), ref_rgb)
        print(
            f"{panel}: page {m.page} scale {m.scale:.5f} px/pt ({m.scale * 72:.2f} dpi)"
            f" box_pt {entry['box_pt']} ncc {m.score:.4f} corr {corr:.4f} mad {mad:.2f}"
        )
        if args.write:
            old = source["panels"].get(panel, {})
            source["panels"][panel] = {**{k: v for k, v in old.items() if k not in entry}, **entry}
            changed = True
    if changed:
        save_spec(spec)
        print(f"wrote {SPEC_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--papers-dir",
            help=f"folder with the PDFs, searched first (then ${PAPERS_DIR_ENV}, <notes_dir>/papers/, reference-material/papers/)",
        )
        p.add_argument("--panel", nargs="+", metavar="PAPER/PANEL", help="only these panels, e.g. mehregan/4a")
        p.add_argument("--arxiv", action="store_true", help="use the arXiv source (Ravivarapu, CC BY 4.0) instead of the publisher PDFs")
        p.add_argument("--renderer", choices=("pdfium", "pdftoppm"), help="force a renderer (default: pypdfium2 if installed)")

    p = sub.add_parser("build", help="write docs/controllers/<controller>/figures/images/<panel>/paper.png (or arxiv_v1.png with --arxiv)")
    common(p)
    p.add_argument("--out-root", help="write under this folder instead of the repo (same figures/... layout)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify", help="compare freshly rendered crops with existing ones")
    common(p)
    p.add_argument("--reference-root", help="folder holding docs/controllers/<controller>/figures/images/<panel>/<name> (default: the repo)")
    p.add_argument("--reference-name", help="reference file name (default: the source's output name)")
    p.add_argument("--min-corr", type=float, default=0.95)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("calibrate", help="locate panels in the PDF by matching reference crops; --write updates the spec")
    common(p)
    p.add_argument("--reference-root", help="folder holding docs/controllers/<controller>/figures/images/<panel>/<name> (default: the repo)")
    p.add_argument("--reference-name", help="reference crop file name (default: paper.png)")
    p.add_argument("--page", type=int, help="search only this page (1-based)")
    p.add_argument("--refine", action="store_true", help="skip the coarse scan; start from the spec's current box")
    p.add_argument("--write", action="store_true", help="store the result in paper_crops.json")
    p.set_defaults(func=cmd_calibrate)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except CropError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
