"""Refresh ``docs/figures/paper_1.md`` after paper figure plot scripts run.

Plot scripts write replication PNGs under ``figures/papers/`` and JSON caches under
``artifacts/figures/papers/``. This module only updates caption markers in the comparison doc.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_1_DOC = REPO_ROOT / "docs" / "figures" / "paper_1.md"

PAPER_1B_PNG = "figures/papers/1/1b/gpi_psd.png"
PAPER_1B_MANIFEST = "artifacts/figures/papers/1/1b/manifest.json"
PAPER_2A_PNG = "figures/papers/1/2a/beta_power.png"
PAPER_2A_MANIFEST = "artifacts/figures/papers/1/2a/manifest.json"
PAPER_1B_REF = "figures/papers/1/1b/paper.png"
PAPER_2A_REF = "figures/papers/1/2a/paper.png"


def _today() -> str:
    return date.today().isoformat()


def _caption_1b(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    if not seeds:
        seed_note = "see manifest"
    elif len(seeds) == 1:
        seed_note = f"seed {seeds[0]}"
    else:
        seed_note = f"seeds {seeds[0]}–{seeds[-1]} mean"
    duration = manifest.get("duration_s", 10)
    return f"{seed_note}, {duration:.0f} s segment ({_today()})"


def _caption_2a(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    seed_note = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {','.join(map(str, seeds))}"
    sampling = manifest.get("sampling", "trailing")
    if sampling == "segment":
        seg = manifest.get("segment_s", 2)
        protocol = f"{seg:.0f} s whole-segment bins"
    else:
        step = manifest.get("step_s", 0.2)
        window = manifest.get("window_s", 2.0)
        warmup = manifest.get("warmup_s")
        if warmup:
            protocol = (
                f"14 s sim (2 s pre-roll), plot = sim − 2 s, "
                f"{step:g} s trailing / {window:g} s window (end sim 14 s)"
            )
        else:
            protocol = f"{step:g} s trailing / {window:g} s window"
    return f"{protocol}, {seed_note} ({_today()})"


def _replace_marker(text: str, marker: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}:start -->)(.*?)(<!-- {marker}:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        msg = f"missing marker block {marker} in {PAPER_1_DOC}"
        raise ValueError(msg)
    return pattern.sub(rf"\1\n{body}\n\3", text, count=1)


def _default_paper_1_doc(*, caption_1b: str, caption_2a: str) -> str:
    return f"""# Mehregan et al. (paper 1) — figure comparisons

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/papers/`; JSON caches to `artifacts/figures/papers/`.

| Panel | Script | Spec |
|-------|--------|------|
| Fig 1b — GPi PSD | `scripts/figures/papers/1/1b/plot.py` | [plant.md](../plant.md) |
| Fig 2a — GPi $P_\\beta$ time series | `scripts/figures/papers/1/2a/plot.py` | [plant.md](../plant.md) |

See also [current-figures.md](../../current-figures.md) for run commands and paths.

---

## Fig 1b — GPi PSD

Mean GPi multitaper power spectral density (1–50 Hz) for three conditions: **healthy control**, **PD no treatment**, and **PD + 130 Hz STN cDBS**. In the paper panel, untreated PD shows a sharp beta-band peak (~12–13 Hz) well above healthy; 130 Hz cDBS suppresses that peak while leaving a smaller secondary bump at higher frequency. Our replication should preserve the ordering **PD > healthy** on beta power and **130 Hz cDBS < PD** (see [plant.md](../plant.md) and seeds-averaged runs in [current-figures.md](../../current-figures.md)).

### Paper (Mehregan et al.)

![Paper Fig 1b](../../{PAPER_1B_REF})

### Replication

![Replication Fig 1b](../../{PAPER_1B_PNG})

<!-- caption-1b:start -->
**Caption:** {caption_1b}

**Manifest:** `{PAPER_1B_MANIFEST}`
<!-- caption-1b:end -->

---

## Fig 2a — GPi $P_\\beta$ time series

GPi beta-band power ($P_\\beta$, Eq. 1, 13–35 Hz) over **12 s**: **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Both traces share a high, overlapping baseline for **0–2 s** (no STN stimulation). A dashed vertical at **2 s** marks cDBS onset for the blue trace only. After onset, blue falls quickly to a low plateau; red stays elevated. Dense continuous lines (not 2 s step bins).

### Paper (Mehregan et al.)

![Paper Fig 2a](../../{PAPER_2A_REF})

### Replication

![Replication Fig 2a](../../{PAPER_2A_PNG})

<!-- caption-2a:start -->
**Caption:** {caption_2a}

**Manifest:** `{PAPER_2A_MANIFEST}`
<!-- caption-2a:end -->

### Side-by-side checklist

Qualitative gates first; numeric bands are approximate (paper read from panel; replication from `{PAPER_2A_MANIFEST}`, seed 0).

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Plot style** | Dense line trace, 0–12 s | Dense trailing line (`--sampling trailing`, 61 points) | ✓ |
| **Axes** | Time (sec) 0–12; y **PSD** ~100–600 | Same labels and range | ✓ |
| **Legend** | PD no Treatment; PD 130 Hz Treatment | Same labels and colors | ✓ |
| **DBS onset marker** | Dashed vertical at **2 s** | Dashed vertical at 2 s | ✓ |
| **0–2 s overlap** | Red and blue identical | Identical (max Δ = 0) | ✓ |
| **$t=0$ baseline** | ~460–470 (high, not zero) | ~513 | ✓ (same band) |
| **$t=2$ level** | ~480 | ~472 | ✓ |
| **Blue drop after 2 s** | Sharp fall; ~300 by $t \\approx 3$ | ~298 at $t=3$ | ✓ |
| **Blue floor** | ~160–170 by $t \\approx 4$; ripple ~160–210 | ~150 at $t=4$; ~160–170 for $t=4$–12 | ✓ |
| **Red after 2 s** | Stays high ~430–500; no downward trend | High ~440–530; wiggles through $t=12$ | ✓ |
| **Separation after onset** | Blue clearly below red | Large gap from $t \\approx 3$ onward | ✓ |
| **$t=12$ red** | ~435 (still high) | **~503** (window sim `[12, 14]`) | ~✓ (slightly high) |
| **$t=12$ blue** | ~185 (stable low floor) | **~160** (ripple, not flat) | ~✓ (still ~25 below paper) |
| **End behavior** | Both traces wiggle at $t=12$ | Sliding windows through sim 14; no flat tail | ✓ |

**Protocol (2026-07-11):** trailing windows end at sim **14 s**; Fig 2a alone passes enlarged GPI spike buffer (904). Re-run: `uv run python scripts/figures/papers/1/2a/plot.py`.

**Remaining gaps:** blue floor ~25 below paper at $t=12$; single seed (0).
"""


def _caption_block(caption: str, manifest_path: str) -> str:
    return f"**Caption:** {caption}\n\n**Manifest:** `{manifest_path}`"


def _read_caption_from_doc(marker: str) -> str | None:
    if not PAPER_1_DOC.exists():
        return None
    text = PAPER_1_DOC.read_text()
    pattern = re.compile(
        rf"<!-- {marker}:start -->\n\*\*Caption:\*\* (.*?)\n\n\*\*Manifest:",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _ensure_paper_1_doc(*, caption_1b: str | None, caption_2a: str | None) -> None:
    cap_1b = caption_1b or _read_caption_from_doc("caption-1b") or "see manifest"
    cap_2a = caption_2a or _read_caption_from_doc("caption-2a") or "see manifest"
    if not PAPER_1_DOC.exists():
        PAPER_1_DOC.parent.mkdir(parents=True, exist_ok=True)
        PAPER_1_DOC.write_text(_default_paper_1_doc(caption_1b=cap_1b, caption_2a=cap_2a))
        return
    text = PAPER_1_DOC.read_text()
    if caption_1b is not None:
        text = _replace_marker(
            text,
            "caption-1b",
            _caption_block(caption_1b, PAPER_1B_MANIFEST),
        )
    if caption_2a is not None:
        text = _replace_marker(
            text,
            "caption-2a",
            _caption_block(caption_2a, PAPER_2A_MANIFEST),
        )
    PAPER_1_DOC.write_text(text)


def promote_1b(
    *,
    manifest: dict[str, Any],
    curves_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 1b caption in ``docs/figures/paper_1.md``."""
    caption = _caption_1b(manifest)
    if update_docs:
        _ensure_paper_1_doc(caption_1b=caption, caption_2a=None)
    return {
        "png": str(png_path),
        "manifest": PAPER_1B_MANIFEST,
        "curves": str(curves_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_2a(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 2a caption in ``docs/figures/paper_1.md``."""
    caption = _caption_2a(manifest)
    if update_docs:
        _ensure_paper_1_doc(caption_1b=None, caption_2a=caption)
    return {
        "png": str(png_path),
        "manifest": str(PAPER_2A_MANIFEST),
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }
