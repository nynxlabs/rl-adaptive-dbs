# Diagrams

Structure diagrams for the README and docs. Each `<name>.mmd` is a [Mermaid](https://mermaid.js.org) source; `<name>.svg` and `<name>.png` next to it are committed renders, so GitHub, the docs, and the project site show the same image without running Mermaid.

| Source | Shows | Used in |
|--------|-------|---------|
| [pipeline.mmd](pipeline.mmd) | How a panel is checked: paper → spec + digitized curves → plant → training → plot → gates → tracker | README, "How we check" |
| [loop.mmd](loop.mmd) | The shared model: controller ↔ paper adapter ↔ Kumaravelu plant | README, "The shared model" |

## Render

```bash
cd scripts/diagrams
npm ci
node render.mjs          # rewrite every docs/diagrams/<name>.svg + .png
node render.mjs --check  # exit 1 if a committed render is stale (what CI runs)
```

The renderer ([scripts/diagrams/render.mjs](../../scripts/diagrams/render.mjs)) uses [Pele](https://pele.run) for layout and [resvg](https://github.com/thx/resvg-js) for the PNG, both pinned in `scripts/diagrams/package.json`. It needs DejaVu Sans (`DejaVuSans.ttf` and `DejaVuSans-Bold.ttf`; Debian/Ubuntu package `fonts-dejavu-core`), found in the usual font folders or `DIAGRAM_FONT_DIR`. PNG bytes depend on the font files, so `--check` is exact on the CI image (Ubuntu 24.04) and may report a stale PNG on a machine with a different DejaVu build; the SVG doesn't depend on fonts.

Link the **PNG** from markdown. The SVG names DejaVu Sans, and a viewer without that font substitutes one with different widths, so labels can overflow their boxes.

## Style

One image has to read on light and dark pages (GitHub light `#ffffff`, dark `#0d1117`), so renders have a transparent background, mid-grey (`#808080`) edges and edge labels, and edges cut out under their labels. Node colors come from `classDef` lines in each source, using the repo palette (the matplotlib tab10 hues the panel plots use):

| Role | Fill | Text |
|------|------|------|
| Inputs (paper, spec, digitized curves) | `#1f77b4` | white |
| Plant | `#2ca02c` | white |
| Training and plots | `#ff7f0e` | `#111111` |
| Gates, reward, adapters | `#9467bd` | white |
| Tracker | `#17becf` | `#111111` |

Orange and cyan are light enough that white text on them falls under 3:1 contrast, so their text is near-black.
