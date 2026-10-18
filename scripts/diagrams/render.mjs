// Render docs/diagrams/*.mmd (Mermaid) to a transparent SVG + PNG pair with Pele and resvg.
//
//   cd scripts/diagrams && npm ci
//   node render.mjs            # write docs/diagrams/<name>.svg and <name>.png
//   node render.mjs --check    # exit 1 if any committed SVG/PNG differs from a fresh render
//
// One image has to read on GitHub's light (#ffffff) and dark (#0d1117) pages, on the docs
// site, and anywhere else the README is mirrored, so the output is plain SVG: Pele's CSS
// variables are resolved to literal colors (lines and edge labels in mid-grey), CSS style
// declarations become presentation attributes, and edges are masked out under their labels
// so a line never runs through its own text. Node colors come from each .mmd's classDef.
//
// Fonts: PNGs are rendered with DejaVu Sans only (no system font scan; the scan costs ~1 s
// per image), found in DIAGRAM_FONT_DIR or the usual system font folders. The --check run
// in CI uses the same DejaVu files (Ubuntu fonts-dejavu-core), so PNG bytes match.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Resvg } from '@resvg/resvg-js';
import { render } from 'pele';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const DIAGRAMS = path.join(REPO, 'docs', 'diagrams');

// Mid-grey reads on both GitHub backgrounds (about 3.9:1 on white, 5.3:1 on #0d1117).
const NEUTRAL = '#808080';
const PALETTE = {
  bg: 'none',
  fg: NEUTRAL,
  muted: NEUTRAL,
  line: NEUTRAL,
  surface: NEUTRAL,
  'surface-alt': NEUTRAL,
  border: NEUTRAL,
  accent: NEUTRAL,
};
const FONT_FAMILY = 'DejaVu Sans';
const PNG_ZOOM = 2;
// Extra space (px) between an edge label and the cut ends of its line.
const LABEL_GAP = 4;
const PELE_OPTIONS = { fontFamily: FONT_FAMILY, responsive: false, padding: 16, autoDirection: false };

// CSS properties that SVG also accepts as presentation attributes; anything else in a style
// attribute (custom properties, letter-spacing: normal, ...) is dropped.
const PRESENTATION = new Set([
  'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray',
  'stroke-linecap', 'stroke-linejoin', 'opacity', 'color', 'font-family', 'font-size',
  'font-weight', 'font-style', 'text-anchor', 'dominant-baseline', 'rx', 'ry',
]);

function findFonts() {
  const names = ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'];
  const dirs = [
    process.env.DIAGRAM_FONT_DIR,
    '/usr/share/fonts/truetype/dejavu',
    '/usr/share/fonts/TTF',
    '/usr/share/fonts/dejavu',
    '/usr/share/fonts/dejavu-sans-fonts',
    '/opt/homebrew/share/fonts',
    '/usr/local/share/fonts',
    path.join(os.homedir(), 'Library', 'Fonts'),
    '/Library/Fonts',
    'C:\\Windows\\Fonts',
  ].filter(Boolean);
  for (const dir of dirs) {
    const files = names.map((n) => path.join(dir, n));
    if (files.every((f) => fs.existsSync(f))) return files;
  }
  throw new Error(
    `DejaVu Sans not found (${names.join(', ')}). Install it (Debian/Ubuntu: fonts-dejavu-core) ` +
      'or set DIAGRAM_FONT_DIR to the folder that holds both files.',
  );
}

function resolveVars(svg) {
  let prev;
  // var(--pele-x, fallback) -> palette value, or the fallback Pele wrote in.
  const pele = /var\(--pele-([a-z0-9-]+),\s*([^()]*?(?:\([^()]*\))?[^()]*?)\)/g;
  do {
    prev = svg;
    svg = svg.replace(pele, (_, key, fallback) => PALETTE[key] ?? fallback.trim());
  } while (svg !== prev);
  // Pele's internal --_x properties are declared once on the root <svg>.
  const internal = {};
  for (const m of svg.matchAll(/(--_[a-z0-9]+):([^;"]+)/g)) if (!(m[1] in internal)) internal[m[1]] = m[2].trim();
  do {
    prev = svg;
    svg = svg.replace(/var\((--_[a-z0-9]+)\)/g, (m, key) => internal[key] ?? m);
  } while (svg !== prev);
  if (svg.includes('var(')) throw new Error('unresolved CSS variable left in SVG');
  return svg;
}

function parseAttrs(text) {
  const attrs = new Map();
  for (const m of text.matchAll(/([A-Za-z_:][\w:.-]*)="([^"]*)"/g)) attrs.set(m[1], m[2]);
  return attrs;
}

function styleToAttrs(attrs) {
  const style = attrs.get('style');
  if (style === undefined) return attrs;
  attrs.delete('style');
  for (const decl of style.split(';')) {
    const i = decl.indexOf(':');
    if (i < 0) continue;
    const prop = decl.slice(0, i).trim();
    let value = decl.slice(i + 1).trim();
    if (prop === 'font') {
      // "16px sans-serif" -> font-size (a later font-family declaration wins for the family).
      const size = value.match(/(\d+(?:\.\d+)?px)/);
      if (size) attrs.set('font-size', size[1]);
      continue;
    }
    if (!PRESENTATION.has(prop)) continue;
    if ((prop === 'rx' || prop === 'ry') && value.endsWith('px')) value = value.slice(0, -2);
    attrs.set(prop, value); // CSS beats presentation attributes, so the style value wins.
  }
  return attrs;
}

function flattenStyles(svg) {
  return svg.replace(/<([A-Za-z][\w:-]*)(\s[^<>]*?)?(\/?)>/g, (m, tag, rest = '', close) => {
    if (!rest.includes('style="') && !rest.includes('"transparent"')) return m;
    const attrs = styleToAttrs(parseAttrs(rest));
    for (const [k, v] of attrs) if (v === 'transparent') attrs.set(k, 'none');
    const out = [...attrs].map(([k, v]) => ` ${k}="${v}"`).join('');
    return `<${tag}${out}${close}>`;
  });
}

// Hide edge lines under edge labels: a mask that is white everywhere except black boxes
// where the labels sit. The labels keep a transparent background.
function maskEdgesUnderLabels(svg) {
  const labels = svg.match(/<g class="pele-edge-labels"[\s\S]*?<\/g><\/g>/);
  if (!labels) return svg;
  const boxes = [...labels[0].matchAll(/<rect ([^>]*?)\/>/g)].map((m) => parseAttrs(m[1]));
  if (!boxes.length) return svg;
  const view = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
  const cuts = boxes
    .map((a) => {
      const [x, w] = [Number(a.get('x')) - LABEL_GAP, Number(a.get('width')) + 2 * LABEL_GAP];
      return `<rect x="${+x.toFixed(2)}" y="${a.get('y')}" width="${+w.toFixed(2)}" height="${a.get('height')}" fill="#000000"/>`;
    })
    .join('');
  const mask =
    `<defs><mask id="edge-label-cut" maskUnits="userSpaceOnUse" x="${view[0]}" y="${view[1]}" width="${view[2]}" height="${view[3]}">` +
    `<rect x="${view[0]}" y="${view[1]}" width="${view[2]}" height="${view[3]}" fill="#ffffff"/>${cuts}</mask></defs>`;
  return svg
    .replace(/(<svg[^>]*>)/, `$1${mask}`)
    .replace('<g class="pele-edges"', '<g class="pele-edges" mask="url(#edge-label-cut)"');
}

function renderDiagram(source, fonts) {
  const { svg: raw } = render(source, PELE_OPTIONS);
  const svg = maskEdgesUnderLabels(flattenStyles(resolveVars(raw))) + '\n';
  const png = new Resvg(svg, {
    fitTo: { mode: 'zoom', value: PNG_ZOOM },
    font: { loadSystemFonts: false, fontFiles: fonts, defaultFontFamily: FONT_FAMILY },
  })
    .render()
    .asPng();
  return { svg: Buffer.from(svg), png: Buffer.from(png) };
}

function main() {
  const check = process.argv.includes('--check');
  const fonts = findFonts();
  const sources = fs.readdirSync(DIAGRAMS).filter((f) => f.endsWith('.mmd')).sort();
  if (!sources.length) throw new Error(`no .mmd files in ${DIAGRAMS}`);
  const stale = [];
  for (const file of sources) {
    const name = file.slice(0, -4);
    const out = renderDiagram(fs.readFileSync(path.join(DIAGRAMS, file), 'utf8'), fonts);
    for (const [ext, bytes] of Object.entries(out)) {
      const target = path.join(DIAGRAMS, `${name}.${ext}`);
      const rel = path.relative(REPO, target);
      if (check) {
        if (!fs.existsSync(target) || !fs.readFileSync(target).equals(bytes)) stale.push(rel);
      } else {
        fs.writeFileSync(target, bytes);
        console.log(`wrote ${rel} (${bytes.length} bytes)`);
      }
    }
  }
  if (check) {
    if (stale.length) {
      console.error(`Stale diagram renders (re-run: cd scripts/diagrams && npm ci && node render.mjs):`);
      for (const rel of stale) console.error(`  ${rel}`);
      process.exit(1);
    }
    console.log(`${sources.length} diagrams up to date`);
  }
}

main();
