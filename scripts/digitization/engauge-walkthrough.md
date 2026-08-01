# Engauge Digitizer walkthrough

Offline GUI digitizer for paper curves. Use it for a **rough auto pass**
(color filter + Segment Fill), then clean up a few points by hand.
Exports feed the same `curves.json` schema as WebPlotDigitizer via
`normalize_engauge.py`.

**Default gate anchors stay WPD-validated.** Engauge is an alternate
ingest path — great when you prefer a desktop app or grids fight WPD.

## Install (this machine)

```bash
sudo apt-get install -y engauge-digitizer engauge-digitizer-doc
```

Binary: `/usr/bin/engauge`. Manual pages: `man engauge`.

### WSL / WSLg display

Engauge is a Qt GUI. From WSL:

```bash
export DISPLAY=:0
engauge
```

Optional: open a panel image immediately:

```bash
export DISPLAY=:0
engauge -import figures/mehregan/images/1b/paper.png
```

If the window does not appear, confirm WSLg is healthy (`/mnt/wslg`
exists) and that Windows graphics drivers are up to date. Re-try from a
terminal inside Windows Terminal / Cursor attached to WSL while you are
at the machine.

## Digitize one panel (HITL)

1. **Import** the panel PNG (`File → Import`, or `-import` on the CLI).
2. **Axis points** — place three axis points (origin, x-max, y-max) and
   type the tick values. Get the **y-scale right first** (raw PSD vs
   PSD×10³ vs Error Index). Wrong units are the #1 digitization bug.
3. **Name curves** — `Settings → Curves` / Curves list: one named curve
   per series (e.g. `PD no Treatment`, `Healthy Control`,
   `PD 130 Hz Treatment`).
4. **Color filter** — `Settings → Filter` (Color Filter). Drag foreground
   sliders until only the target series remains in the preview. Grid /
   legend pollution means the filter is too loose.
5. **Segment Fill** (or Point Match) — click the filtered curve; Engauge
   auto-traces. Delete outliers, add gaps by hand.
6. **Next series** — refilter for the next color, select that curve name,
   Segment Fill again.
7. **Save project** — `File → Save` →
   `artifacts/figures/papers/<paper>/<panel>/paper_digitization/<panel>.dig`
8. **Export CSV** — `Settings → Export Setup` recommended for this repo:
   - **Raw Xs and Ys**
   - **All curves on each line** (or shared-x multi-column — both are
     accepted by `normalize_engauge.py`)
   - Then `File → Export` →
     `.../paper_digitization/<panel>_engauge.csv`

## Normalize into the shared schema

```bash
uv run python scripts/digitization/normalize_engauge.py \
    --input artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_engauge.csv \
    --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_engauge.json \
    --figure mehregan_fig1b
```

Compare against a WPD (or PIL) schema file:

```bash
uv run python scripts/digitization/compare_curves.py \
    --ref artifacts/figures/papers/mehregan/1b/paper_digitization/curves_wpd.json \
    --hyp artifacts/figures/papers/mehregan/1b/paper_digitization/curves_pil.json \
    --json artifacts/figures/papers/mehregan/1b/paper_digitization/compare_pil_vs_wpd.json
```

## Export validation checklist

| check | pass if |
|-------|---------|
| Structure | one series per condition; dozens of points per curve |
| X span | extracted x covers the paper x-axis |
| Y scale | peaks sit inside the paper y tick range |
| Ordering | matches the paper claim (e.g. PD > healthy, 130 Hz < PD) |
| vs WPD | same ordering; RMSE not wildly off on a shared panel |

## Artifact layout

```
artifacts/figures/papers/<paper>/<panel>/paper_digitization/
  <panel>.dig              # Engauge project (reproducible)
  <panel>_engauge.csv      # export
  curves_engauge.json      # normalized schema
  # alongside WPD when present:
  <panel>.wpd.json
  *_digitized.csv
  curves_wpd.json
```

## CLI cheat sheet

| flag | meaning |
|------|---------|
| `-import FILE` | open an image at startup |
| `-open FILE` | open a `.dig` project |
| `-axes XMIN XMAX YMIN YMAX` | auto-place axis points (still review!) |
| `-export FILE` | export active doc on shutdown |
| `-reset` | factory-reset Engauge settings |
