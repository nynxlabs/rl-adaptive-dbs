# digitization pipeline

Reusable paper-curve digitization for figure-quantification gates.
Multiple extraction routes, one output schema.

## layout

| file | purpose |
|------|---------|
| `schema.py` | thin series record: `n`, full `xy`, optional `color_rgba` (no convenience stats) |
| `extract_pil.py` | **automated** color-mask tracing from a paper PNG (seconds, repeatable) |
| `normalize_wpd.py` | **manual / HITL** WebPlotDigitizer export → same schema (WPD project JSON or CSV) |
| `normalize_engauge.py` | **manual / HITL** Engauge Digitizer CSV → same schema |
| `compare_curves.py` | compare two schema JSON files (RMSE / Pearson / ordering gates) |
| `pil_to_wpd.py` | PIL rough extract → WPD project JSON (editable seed in the browser) |
| `engauge-walkthrough.md` | install, WSLg, color-filter HITL steps |
| `figs/` | per-figure configs: axis box, tick calibration, series color masks |

## workflow

1. **fast hypothesis:** run the PIL extractor for a figure config.
2. **HITL digitize:** WebPlotDigitizer (default) or Engauge (desktop) —
   auto-extract, then clean a few points. See `engauge-walkthrough.md`.
3. **normalize** the export into the shared schema (`normalize_wpd.py` /
   `normalize_engauge.py`).
4. **compare:** `compare_curves.py` (PIL vs WPD, Engauge vs WPD, …) and/or
   the panel's `compare_paper.py` against the replication curves cache.

**Gate anchors:** prefer a human-validated WPD (or Engauge) export. An
automated PIL trace is a hypothesis, not ground truth.

## usage

```bash
# automated PIL extraction
uv run python scripts/digitization/extract_pil.py \
    --png figures/mehregan/images/1b/paper.png \
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
    --png figures/mehregan/images/1b/paper.png \
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
