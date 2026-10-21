# Diagrams

Structure diagrams for the README and docs. Each `<name>.mmd` is a [Mermaid](https://mermaid.js.org) source; `<name>.svg` and `<name>.png` next to it are committed renders, so GitHub, the docs, and the project site show the same image without running Mermaid.

| Source | Shows | Used in |
|--------|-------|---------|
| [pipeline.mmd](pipeline.mmd) | How a panel is checked: paper → spec + digitized curves → plant → training → plot → gates → figure page + tracker | README, "How we check" |
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

One image has to read on light and dark pages (GitHub light `#ffffff`, dark `#0d1117`), so renders have a transparent background, mid-grey (`#808080`) edges and edge labels, and edges cut out under their labels. Node colors come from `classDef` lines in each source, using the pastel palette the panel plots use (Catppuccin Frappé):

| Role | Fill | Text |
|------|------|------|
| Inputs (paper, spec, digitized curves) | `#8caaee` (blue) | `#232634` |
| Plant | `#a6d189` (green) | `#232634` |
| Training and plots | `#ef9f76` (peach) | `#232634` |
| Gates, reward, adapters | `#ca9ee6` (mauve) | `#232634` |
| Figure page + tracker | `#81c8be` (teal) | `#232634` |

The pastels are light, so every node uses the same dark text for contrast.
