# digitization pipeline

Reusable paper-curve digitization for figure-quantification gates.
Two extraction routes, one output schema.

## layout

| file | purpose |
|------|---------|
| `schema.py` | shared series stats (`n`, `start`/`end`, `early_mean`/`late_mean`, `drop_early_to_late`, `min`/`max`/`mean`, `at`, `slope`) |
| `extract_pil.py` | **automated** color-mask tracing from a paper PNG (seconds, repeatable) |
| `normalize_wpd.py` | **manual** WebPlotDigitizer export → same schema (WPD project JSON or CSV) |
| `figs/` | per-figure configs: axis box, tick calibration, series color masks |

## workflow

1. **fast hypothesis:** run the PIL extractor for a figure config.
2. **validate:** WebPlotDigitizer spot-check on one panel (see the
   `webplotdigitizer-walkthrough.md` skill reference for the export
   validation checklist).
3. **compare:** run the panel's `compare_paper.py` against the
   replication curves cache to get RMSE / Pearson r / beta stats /
   ordering gates.

## usage

```bash
# automated PIL extraction
uv run python scripts/digitization/extract_pil.py \
    --png figures/mehregan/images/1b/paper.png \
    --config scripts/digitization/figs/mehregan_fig1b.json \
    --out /tmp/curves_pil.json

# normalize a WPD project JSON export
uv run python scripts/digitization/normalize_wpd.py \
    --input artifacts/figures/papers/mehregan/1b/paper_digitization/mehregan_fig_1b.wpd.json \
    --out /tmp/curves_wpd.json

# normalize a WPD CSV export (with optional series-name map)
uv run python scripts/digitization/normalize_wpd.py \
    --input artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_paper_digitized.csv \
    --series-map 'PD no Treatment=pd,Healthy Control=healthy,PD 130 Hz Treatment=pd_130hz' \
    --out /tmp/curves_wpd.json
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
  series (`/tmp` debug script pattern in the skill reference).
- an automated trace is a **hypothesis, not ground truth**. validate at
  least one panel per figure with WPD before using it as a gate anchor.
- keep the artifact next to the panel manifest, and record provenance
  (see `digitization-schema.md` provenance check).
