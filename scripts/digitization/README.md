# digitization pipeline

Reusable paper-curve digitization for figure-quantification gates.
Multiple extraction routes, one output schema.

## layout

| file | purpose |
|------|---------|
| `schema.py` | thin series record: `n`, full `xy`, optional `color_rgba` (no convenience stats) |
| `paper_gates.py` | **Mehregan Paper 1 gate builders** — load refined JSON, x-window means, ordering / ratio gates wired from panel `plot.py` |
| `nguyen_gates.py` | **Nguyen Paper 2 gate builders** — load ``curves_fig*.json``, x-window means / ratios for panels 3–7 |
| `extract_pil.py` | **automated** color-mask tracing from a paper PNG (seconds, repeatable) |
| `normalize_wpd.py` | **manual / HITL** WebPlotDigitizer export → same schema (WPD project JSON or CSV) |
| `normalize_engauge.py` | **manual / HITL** Engauge Digitizer CSV → same schema |
| `compare_curves.py` | compare two schema JSON files (RMSE / Pearson / ordering gates) |
| `pil_to_wpd.py` | PIL rough extract → WPD project JSON (editable seed in the browser) |
| `engauge-walkthrough.md` | install, WSLg, color-filter HITL steps |
| `figs/` | per-figure configs: axis box, tick calibration, series color masks |
| `paper_crops.py` + `paper_crops.json` | builds the local-only paper crops (`paper.png`) from your own copy of each paper |

## paper crops (local only)

The tools here read the published figure as
`docs/controllers/<controller>/figures/images/<panel>/paper.png`. Those crops are the publishers'
figures, so they are **not in the repository** (gitignored) — build them from
your own copy of each paper:

1. Save the PDFs as `reference-material/papers/mehregan.pdf`, `nguyen.pdf`
   and `ravivarapu.pdf` (gitignored; `paper_1.pdf`, `paper_2.pdf`,
   `paper_3.pdf` work too). The script also looks in `--papers-dir`, then
   `RL_DBS_PAPERS_DIR`, then `<notes_dir>/papers/` (`export.notes_dir` /
   `RL_DBS_NOTES_DIR`, [docs/guides/cli.md](../../docs/guides/cli.md) §5.6), before that
   default folder.
2. Build all crops, or a few:

   ```bash
   uv run --group papers python scripts/digitization/paper_crops.py build
   uv run --group papers python scripts/digitization/paper_crops.py build --panel mehregan/4a nguyen/6
   ```

`paper_crops.json` holds each panel's page, crop box (PDF points, from the
top-left corner) and pixel size, so a built crop has the exact pixel geometry
the `figs/*.json` boxes were measured on. A PDF whose checksum differs from
the spec's (another download of the same paper) gives a warning, not an
error; `paper_crops.py verify --reference-root <folder>` compares built crops
against existing ones. Rendering uses pypdfium2 from the `papers` dependency
group (Windows, macOS, Linux), else Poppler's `pdftoppm` if it is on PATH.
`calibrate` re-derives a panel's box by matching a reference crop against
the rendered pages; you only need it to add a panel.

Ravivarapu et al. is also on arXiv under CC BY 4.0 (arXiv:2507.06326v1). Its
six panels are committed as `docs/controllers/sea-dbs/figures/images/<panel>/arxiv_v1.png`;
`paper_crops.py build --arxiv` rebuilds them, downloading the arXiv PDF into
the papers folder if it is missing.

## workflow

1. **fast hypothesis:** run the PIL extractor for a figure config.
2. **HITL digitize:** WebPlotDigitizer (default) or Engauge (desktop) —
   auto-extract, then clean a few points. See `engauge-walkthrough.md`.
3. **normalize** the export into the shared schema (`normalize_wpd.py` /
   `normalize_engauge.py`).
4. **compare / gate:** `compare_curves.py` (PIL vs WPD, …) and panel
   `plot.py` gates via `paper_gates.py` (Mehregan) or `nguyen_gates.py`
   (Nguyen) against normalized curves JSON under each paper's
   ``paper_digitization/`` tree.

**Gate anchors:** prefer a human-validated WPD (or Engauge) export. An
automated PIL trace is a hypothesis, not ground truth. Paper figures are
usually **one seed**; gates use ordering + x-window ratios/drops, not
pointwise wiggle RMSE.

## usage

```bash
# automated PIL extraction
uv run python scripts/digitization/extract_pil.py \
    --png docs/controllers/ddpg/figures/images/1b/paper.png \
    --config scripts/digitization/figs/mehregan_fig1b.json \
    --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_pil.json

# normalize a WPD project JSON export
uv run python scripts/digitization/normalize_wpd.py \
    --input artifacts/figures/papers/mehregan/1b/paper_digitization/mehregan_fig_1b.wpd.json \
    --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_wpd.json \
    --figure mehregan_fig1b

# normalize an Engauge CSV export
uv run python scripts/digitization/normalize_engauge.py \
    --input artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_engauge.csv \
    --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_engauge.json \
    --figure mehregan_fig1b

# compare PIL hypothesis vs WPD anchor
uv run python scripts/digitization/compare_curves.py \
    --ref artifacts/figures/papers/mehregan/1b/paper_digitization/curves_wpd.json \
    --hyp artifacts/figures/papers/mehregan/1b/paper_digitization/curves_pil.json \
    --json artifacts/figures/papers/mehregan/1b/paper_digitization/compare_pil_vs_wpd.json

# seed WPD with a PIL rough pass (JSON and/or one-click .tar with image embedded)
uv run python scripts/digitization/pil_to_wpd.py \
    --curves artifacts/figures/papers/mehregan/1b/paper_digitization/curves_pil.json \
    --config scripts/digitization/figs/mehregan_fig1b.json \
    --png docs/controllers/ddpg/figures/images/1b/paper.png \
    --out artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_pil_seed.wpd.json \
    --tar artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_pil_seed.wpd.tar
# then in WPD: File → Load Project → pick the .tar (image + points in one step)
```

## figure configs

A `figs/<figure>.json` describes one panel:

```json
{
  "figure": "mehregan_fig1b",
  "panel": {
    "box": [54, 252, 58, 208],
    "x_range": [0, 50],
    "y_range": [0, 70],
    "n_bins": 120,
    "legend_exclude_frac": 0.18,
    "legend_exclude_top_frac": 0.14,
    "series": {
      "pd": {"r_min": 200, "r_max": 255, "g_min": 130, "g_max": 235, "b_min": 90, "b_max": 200},
      "healthy": {"r_min": 0, "r_max": 70, "g_min": 0, "g_max": 70, "b_min": 0, "b_max": 70},
      "pd_130hz": {"r_min": 0, "r_max": 130, "g_min": 0, "g_max": 130, "b_min": 100, "b_max": 255}
    }
  }
}
```

- `box` = `[top, bottom, left, right]` pixel bounds of the axis frame
  (read from WPD calibration points, shave ~1 px to exclude frame lines)
- `x_range` / `y_range` = tick values (paper y-axis label FIRST: raw
  P_beta vs PSD(x10³) vs Error Index)
- `legend_exclude_top_frac` for legends spanning the top; the corner
  exclusion (`legend_exclude_frac`) covers top-right legends
- series masks: either RGB bounds (`r_min`...) or a `color` + `tol`

## pitfalls

- OCR of tick labels on small PNGs is unreliable — use known tick values.
- frame lines and legend text pollute loose masks. shave the box, add
  the top-band legend exclusion, and check the pixel histogram per
  series.
- an automated trace is a **hypothesis, not ground truth**. validate at
  least one panel per figure with WPD or Engauge before using it as a
  gate anchor.
- keep the artifact next to the panel manifest. Series records are
  ``n`` + ``xy`` (+ ``color_rgba`` when known). **No** early/late/slope
  convenience stats — gates must slice ``xy`` by real x (Hz, sec, …).
  Provenance: ``method`` is ``wpd-*``, ``engauge-csv``, or ``pil-color-mask``.

## redraw digitized panels

`redraw_digitized.py` draws each panel's digitized paper data with matplotlib.
Public comparisons show this redraw instead of a crop of the published figure.
The script never reads or copies the paper image.

```bash
uv run python scripts/digitization/redraw_digitized.py --all
uv run python scripts/digitization/redraw_digitized.py --panel mehregan/4a --panel nguyen/6
```

- input: `curves_*.json` under `--artifacts` (default `artifacts/figures/papers`).
  `artifacts/` is gitignored, so from a worktree pass the main checkout's folder.
- output: `docs/controllers/<controller>/figures/images/<panel>/digitized_v<N>.png` (`--version`, default 1).
  `--out-root` defaults to the repo root (images land in `docs/controllers/<controller>/figures/images/<panel>/`).
- every image has the caption `Digitized from Fig. X of <paper>` and the footer
  `Our redraw of values read off the published figure; not the original figure.`
- axis labels come from each panel's `plot.py`. Axis ranges come from
  `scripts/digitization/figs/<stem>.json`. The y axis widens to fit every digitized
  point. The x axis widens only when a point overshoots by more than 2% of its span.
- legend labels are the series names. The digitized `color_rgba` is used when present.
- a panel with any missing input is skipped with a warning. The exit code is non-zero
  only when nothing was drawn.
- Mehregan 4b, Nguyen 4 and Nguyen 5 are two-subplot figures, Nguyen 6 is a 2×2 grid,
  and Nguyen 3 is a box plot drawn from `curves_fig3_boxplot.json`.
- bump `--version` to refresh an image that a viewer has cached, since the file name
  is the cache key.
- pitfall: the Nguyen Fig 4 reward `Raw` series has readings above the paper's axis
  top (0), at episodes near 470–500. They look like digitization artifacts. They are
  drawn as read, which stretches that axis to about +68,000.
