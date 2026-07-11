# Mehregan et al. (paper 1) — figure comparisons

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/papers/`; JSON caches to `artifacts/figures/papers/`.

| Panel | Script | Spec |
|-------|--------|------|
| Fig 1b — GPi PSD | `scripts/figures/papers/1/1b/plot.py` | [plant.md](../plant.md) |
| Fig 2a — GPi $P_\beta$ time series | `scripts/figures/papers/1/2a/plot.py` | [plant.md](../plant.md) |

See also [current-figures.md](../../current-figures.md) for run commands and paths.

---

## Fig 1b — GPi PSD

Mean GPi multitaper power spectral density (1–50 Hz) for three conditions: **healthy control**, **PD no treatment**, and **PD + 130 Hz STN cDBS**. In the paper panel, untreated PD shows a sharp beta-band peak (~12–13 Hz) well above healthy; 130 Hz cDBS suppresses that peak while leaving a smaller secondary bump at higher frequency. Our replication should preserve the ordering **PD > healthy** on beta power and **130 Hz cDBS < PD** (see [plant.md](../plant.md) and seeds-averaged runs in [current-figures.md](../../current-figures.md)).

### Paper (Mehregan et al.)

![Paper Fig 1b](../../figures/papers/1/1b/paper.png)

### Replication

![Replication Fig 1b](../../figures/papers/1/1b/gpi_psd.png)

<!-- caption-1b:start -->
**Caption:** see manifest

**Manifest:** `artifacts/figures/papers/1/1b/manifest.json`
<!-- caption-1b:end -->

---

## Fig 2a — GPi $P_\beta$ time series

GPi beta-band power ($P_\beta$, Eq. 1, 13–35 Hz) over **12 s**: **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Both traces share a high, overlapping baseline for **0–2 s** (no STN stimulation). A dashed vertical at **2 s** marks cDBS onset for the blue trace only. After onset, blue falls quickly to a low plateau; red stays elevated. Dense continuous lines (not 2 s step bins).

### Paper (Mehregan et al.)

![Paper Fig 2a](../../figures/papers/1/2a/paper.png)

### Replication

![Replication Fig 2a](../../figures/papers/1/2a/beta_power.png)

<!-- caption-2a:start -->
**Caption:** 14 s sim (2 s pre-roll), plot = sim − 2 s, 0.2 s trailing / 2 s window (end sim 14 s), seed 0 (2026-07-11)

**Manifest:** `artifacts/figures/papers/1/2a/manifest.json`
<!-- caption-2a:end -->

### Side-by-side checklist

Qualitative gates first; numeric bands are approximate (paper read from panel; replication from `artifacts/figures/papers/1/2a/manifest.json`, seed 0).

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Plot style** | Dense line trace, 0–12 s | Dense trailing line (`--sampling trailing`, 61 points) | ✓ |
| **Axes** | Time (sec) 0–12; y **PSD** ~100–600 | Same labels and range | ✓ |
| **Legend** | PD no Treatment; PD 130 Hz Treatment | Same labels and colors | ✓ |
| **DBS onset marker** | Dashed vertical at **2 s** | Dashed vertical at 2 s | ✓ |
| **0–2 s overlap** | Red and blue identical | Identical (max Δ = 0) | ✓ |
| **$t=0$ baseline** | ~460–470 (high, not zero) | ~513 | ✓ (same band) |
| **$t=2$ level** | ~480 | ~472 | ✓ |
| **Blue drop after 2 s** | Sharp fall; ~300 by $t \approx 3$ | ~298 at $t=3$ | ✓ |
| **Blue floor** | ~160–170 by $t \approx 4$; ripple ~160–210 | ~150 at $t=4$; ~160–170 for $t=4$–12 | ✓ |
| **Red after 2 s** | Stays high ~430–500; no downward trend | High ~440–530; wiggles through $t=12$ | ✓ |
| **Separation after onset** | Blue clearly below red | Large gap from $t \approx 3$ onward | ✓ |
| **$t=12$ red** | ~435 (still high) | **~503** (window sim `[12, 14]`) | ~✓ (slightly high) |
| **$t=12$ blue** | ~185 (stable low floor) | **~160** (ripple, not flat) | ~✓ (still ~25 below paper) |
| **End behavior** | Both traces wiggle at $t=12$ | Sliding windows through sim 14; no flat tail | ✓ |

**Protocol (2026-07-11):** trailing windows end at sim **14 s** (display $t=12$ → `[12, 14]`). Fig 2a alone passes a larger Numba GPI spike buffer (904) so recording is not truncated before sim 14. Re-run: `uv run python scripts/figures/papers/1/2a/plot.py`.

**Remaining gaps:** blue floor ~25 below paper at $t=12$; single seed (0).
