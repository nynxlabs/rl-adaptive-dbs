# Ravivarapu et al. — figure replication gates

**Primary gate spec** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Ship status, side-by-side PNGs, and run commands live in [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md).

Controller / adapter: [sea_dbs/replication.md](../../controllers/sea_dbs/replication.md). Shared plant: [plant.md](../../plant.md). Schematics (paper Figs 1–3) are **out of scope**. Fig 2 (reward curve of Eq. (7)) is optional polish, not a gate.

**Defaults (paper §V.A / Table I):** seed `0` (paper seed unspecified — lock one seed for gates), **2 ms** RL step × **30** steps/episode × **150** training episodes, $\gamma=0.99$, buffer **8192**, batch **32**, $\alpha_a=5\times10^{-4}$, $\alpha_c=10^{-3}$. Variants: `baseline` (DDPG, no PM/GS) vs `paper` (full SEA-DBS).

| Panel | Script | Status |
|-------|--------|--------|
| Fig 4a — training PSD vs episode | `scripts/figures/papers/ravivarapu/4a/plot.py` | **Pass** — v40 (`shape_pass` + `pass`); lab notes [4a.md](../ravivarapu/4a.md) |
| Fig 4b — training reward vs episode | `scripts/figures/papers/ravivarapu/4b/plot.py` | Open — paired with 4a |
| Fig 5a — inference @ 50 Hz | `scripts/figures/papers/ravivarapu/5a/plot.py` | Open |
| Fig 5b — inference @ 30 Hz | `scripts/figures/papers/ravivarapu/5b/plot.py` | Open |
| Fig 6 — FP16 PTQ @ 50 Hz | `scripts/figures/papers/ravivarapu/6/plot.py` | Open |
| Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS) | `scripts/figures/papers/ravivarapu/7/plot.py` | Open |

**Pause note (2026-07-31):** full trains/probes are paused until gates below are implemented from **paper figure digitization** (color-mask readout of `figures/ravivarapu/images/*/paper.png`). Absolute y values have ~±0.01–0.02 (Fig 4a) / ~±10–15 (Figs 5–7) calibration uncertainty; **ordering, shared starts, relative drops, and cross-panel ratios** are the hard targets.

Digitization cache (local): `artifacts/figures/papers/ravivarapu/paper_digitization/curves.json`.

**Scale note:** Fig 4a y-axis is labeled **(PSD × 10⁻³)** with values ~0.32–0.48. Figs 5–7 use **PSD** on ~300–480. Same biomarker: $0.46$ on Fig 4a ↔ ~$460$ on Figs 5–7.

---

## Paper digitization — readout summary

Color-masked curve extraction from paper crops (axis box + blue/red/magenta/green/black masks). Values are **paper targets for gate design**, not replication means.

### Fig 4a — training PSD (× 10⁻³), episodes 0–150

| Metric | Baseline | SEA-DBS | Paper look |
|--------|----------|---------|------------|
| Start (ep ~0) | ~0.47–0.48 | ~0.46–0.47 | **Shared high onset** (within ~0.02) |
| End (ep ~150) | ~0.36–0.38 | ~0.32–0.35 | SEA lower |
| Early mean (first ~5%) | ~0.46–0.47 | ~0.46–0.47 | Matched start band |
| Late mean (2nd half) | ~0.39–0.40 | ~0.37–0.38 | SEA below Baseline |
| Early→late drop | ~0.06–0.07 | ~0.09–0.10 | SEA **~1.3–1.4×** Baseline drop |
| Polyfit slope (/ep) | ~−5×10⁻⁴ | ~−7 to −8×10⁻⁴ | SEA steeper |
| Mid crossover | — | ~ep 40–50 | After crossover SEA stays below |
| Noise | ±0.01–0.02 | ±0.01–0.02 (larger early) | Both noisy; Baseline **does decline** |

Checkpoint samples (digitized, y-cal ≈ 0.30–0.49):

| Episode | Baseline | SEA-DBS |
|---------|----------|---------|
| 0 | 0.479 | 0.471 |
| 20 | 0.438 | 0.435 |
| 50 | 0.428 | 0.416 |
| 100 | 0.402 | 0.385 |
| 150 | 0.374 | 0.341 |

### Fig 4b — training reward, episodes 0–150

| Metric | Baseline | SEA-DBS | Paper look |
|--------|----------|---------|------------|
| Start | ~−1.1 to −1.5 | ~−1.0 to −1.5 | Both low |
| End | ~−0.05 to −0.10 | ~0.00 | SEA higher (near 0) |
| Late mean (2nd half) | lower | higher | SEA above Baseline |
| Pull-ahead | — | ~ep 40+ | Not “SEA higher in first 10%” — early both climb from floor; SEA **pulls ahead mid-run** |

### Fig 5a — inference PSD @ 50 Hz, steps 0–10

| Step | Baseline | SEA-DBS |
|------|----------|---------|
| 0 | ~462–468 | ~462–468 (shared) |
| 5 | ~402–421 | ~334–364 |
| 10 | ~352–382 | ~308–333 |

SEA early→late drop ≈ **1.7–2×** Baseline. Both decline; SEA steeper with mid plateau ~steps 5–8.

### Fig 5b — inference PSD @ 30 Hz, steps 0–10

| Step | Baseline | SEA-DBS |
|------|----------|---------|
| 0 | ~462–471 | shared |
| 10 | ~398–418 | ~388–408 |

Weaker suppression than 5a; final gap SEA below Baseline only ~10. Cross-panel: **5b end ≫ 5a end** for both variants.

### Fig 6 — four series @ 50 Hz, steps 0–10

Shared start ~462–478. End (digitized): Baseline ~371, Baseline+PTQ ~385, SEA-DBS ~310, SEA-DBS+PTQ ~320. SEA and SEA+PTQ track through mid-run; PTQ more volatile after step ~5 (deep dip then recover). Baseline+PTQ slightly **above** Baseline late.

### Fig 7 — ablation @ 50 Hz, steps 0–10

Shared start ~462–484. End ordering (paper): **SEA-DBS ≪ Baseline ≈ Baseline+PM < Baseline+GS**. GS is the **worst** suppressor (ends ~400+), not a mild Baseline upgrade.

---

## Fig 4a — training PSD vs episode

Average **beta-band PSD** across **training episodes** for **Baseline (DDPG)** vs **SEA-DBS** (§V.B / Fig. 4(a)).

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Series** | Baseline vs SEA-DBS | Missing either series |
| 2 | **Shared start** | Both onset in high untreated band (~0.46–0.48 on paper scale) | Starts diverge by ≫0.05, or one starts already suppressed |
| 3 | **Baseline declines** | Modest noisy downward trend (not flat / frozen ε=1) | Baseline flat, rising, or open-loop locked |
| 4 | **SEA below Baseline (late)** | After ~ep 40–50, SEA stays below Baseline | SEA ≥ Baseline on late half mean |
| 5 | **SEA steeper drop** | SEA early→late drop **>** Baseline drop (~1.3× paper) | SEA drop ≤ Baseline drop |
| 6 | **Episode axis** | 150 episodes | Wrong count |
| 7 | **Paired with Fig 4b** | Same training run | Mismatched lineage |

**Proposed automated mirrors (redesign — replace current 3-bool gates):**

| Key | Rule (paper-faithful) |
|-----|------------------------|
| `shared_start` | $\|p_{\mathrm{early}} - b_{\mathrm{early}}\| < 0.05$ (early = first ~5% episodes, min 3) |
| `start_in_high_band` | both early means in $[0.40, 0.52]$ on Fig-4a scale (or documented equivalent after `observation_scale`) |
| `baseline_declines` | $b_{\mathrm{early}} - b_{\mathrm{late}} > 0.02$ |
| `paper_declines` | $p_{\mathrm{early}} - p_{\mathrm{late}} > 0.04$ |
| `paper_below_baseline_late` | $p_{\mathrm{late}} < b_{\mathrm{late}}$ (late = 2nd half) |
| `paper_steeper_drop` | $(p_{\mathrm{early}}-p_{\mathrm{late}}) > (b_{\mathrm{early}}-b_{\mathrm{late}})$ |
| `late_gap_min` | $b_{\mathrm{late}} - p_{\mathrm{late}} > 0.01$ (paper ~0.02–0.03) |

Do **not** pass a frozen / non-learning Baseline. Do **not** use display hacks to force these levels.

**Stimulation drive convention (v23, passing):** `SEA_DBSEnvAdapter` applies each stim action as a **short burst** — 60 ms of 130 Hz carrier within the 100 ms biomarker step (`dbs_burst_ms=60`) — matching the paper's "short bursts rather than continuously" (Eq. (6)). Continuous 130 Hz on the shared Kumaravelu plant drives GPi beta to ~0.12, far below the paper's Fig 4a late levels; the 60 ms burst gives a ~0.35 full-duty floor so SEA-DBS lands at 0.355 (paper 0.340) and the Baseline at 0.375 (paper 0.368). This is a controller convention (docs/controllers/sea_dbs/replication.md §11), not a display or eval shortcut. All gates pass (v8).

---

## Fig 4b — training reward vs episode

**Episode reward** during the same training run as Fig 4a (§V.B / Fig. 4(b)).

**Qualitative gates:**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Series** | Baseline vs SEA-DBS | Missing either |
| 2 | **SEA higher late** | SEA above Baseline on 2nd half | SEA ≤ Baseline late |
| 3 | **Mid-run pull-ahead** | SEA pulls ahead by ~ep 40+ and stays ahead | Only wins on a tiny tail blip |
| 4 | **Both rise from low start** | Start near floor (~−1.5 to −1.0), climb toward 0 | Flat at 0 from episode 1 |
| 5 | **Paired with Fig 4a** | Same `series.json` / checkpoints | Different train |
| 6 | **Reward–PSD consistency** | Reward ↑ as PSD ↓ | Opposite with no note |

**Proposed automated mirrors:**

| Key | Rule |
|-----|------|
| `paper_above_baseline_late` | $p_{\mathrm{late}} > b_{\mathrm{late}}$ |
| `paper_pull_ahead_mid` | mean(SEA[40:80]) > mean(Baseline[40:80]) (or frac equivalents) |
| `both_rise` | both (late − early) > 0.3 |
| **Remove** | `paper_faster_early_rise` comparing first-10% means — **unfaithful** to paper (both start low; SEA does not clearly lead that early) |

---

## Fig 5a — inference @ 50 Hz

Post-train **inference** at **50 Hz** over **10 steps** (Fig. 5(a)).

**Qualitative gates:**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Carrier** | 50 Hz | Wrong carrier |
| 2 | **Shared step-0** | Both ≈ same PSD at step 0 | Starts diverge |
| 3 | **Both decline** | Baseline and SEA fall over 10 steps | Flat / frozen Baseline |
| 4 | **SEA steeper / lower end** | SEA end ≪ Baseline end; drop ~1.7–2× | SEA ≥ Baseline end or similar drop |
| 5 | **vs Fig 5b** | 50 Hz stronger than 30 Hz | 30 Hz beats 50 Hz |
| 6 | **Checkpoints** | From Fig 4 train | Untrained / mismatched |
| 7 | **Steps 0–5** | Early window near digitized paper (MAE ≤ 0.03) | 4-pulse 50 Hz floor stuck ~0.43 |

**Automated shape gates** (normalized `p_beta_norm`; overlay ÷1000 from paper crop ~300–480): `n_steps_ok` = **11 PSD samples** (t=0 untreated + 10 stim actions); `shared_start`; any net drop on both traces (`baseline_declines` / `paper_declines`); `paper_end_below_baseline`; `paper_steeper_drop`; `carrier_hz_ok`; generous steps 0–5 vs digitization (`early_mae_*` ≤ 0.03, SEA drop 0→5 > 0.05). Digitized 50 Hz 10-step drops (~0.10 Baseline / ~0.15 SEA) are not 10-step polish thresholds. Fig 5a eval burst is 100 ms (five 50 Hz pulses); Fig 4a train stays 62 ms @ 130 Hz.

---

## Fig 5b — inference @ 30 Hz

Same layout at **30 Hz** (Fig. 5(b)).

**Qualitative gates:** weaker suppression than 5a; SEA still slightly below Baseline at end; shared start; both decline modestly.

**Automated shape gates:** same ordering as 5a at 30 Hz, plus `weaker_than_50hz_*` (both variants end above the 50 Hz panel). Digitized 30 Hz drops are ~0.06 Baseline / ~0.07 SEA with a small end gap (~0.01); gates still require the gap and any net decline, not those magnitudes.

---

## Fig 6 — FP16 PTQ @ 50 Hz

**Four series** over **10 steps** @ **50 Hz**: Baseline, Baseline+PTQ(fp16), SEA-DBS, SEA-DBS+PTQ(fp16). QAT out of scope.

**Qualitative gates:**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Four series** | All four present | Missing Baseline+PTQ or SEA+PTQ |
| 2 | **Shared start** | All ~same at step 0 | Divergent onsets |
| 3 | **SEA family below Baseline family** | SEA and SEA+PTQ below both Baselines late | PTQ SEA in Baseline band |
| 4 | **PTQ tracks FP32 (SEA)** | SEA+PTQ follows SEA (allow mid-run PTQ volatility) | Systematic divergence to Baseline |
| 5 | **Baseline+PTQ** | Slightly worse / above Baseline late (paper) | Required only as soft/report unless we harden |

**Proposed automated mirrors:** four keys present; `sea_below_baseline`; `sea_ptq_below_baseline`; `sea_ptq_tracks_fp32` (rel gap threshold, tolerate mid volatility); optional `baseline_ptq_above_or_near_baseline`.

---

## Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS)

**PSD over 10 steps.** Variants: `baseline`, `baseline-pm`, `baseline-gs`, `paper`.

**Qualitative gates:**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Four series** | All four | Missing variant |
| 2 | **SEA lowest** | SEA most suppressed | Any other below SEA late |
| 3 | **GS worst / limited** | Baseline+GS **highest** PSD (least suppression) | +GS matches SEA or beats Baseline strongly downward |
| 4 | **+PM ≈ Baseline class** | +PM near Baseline, not SEA | +PM identical to full SEA |
| 5 | **Shared start + 10 steps** | — | Wrong protocol |

**Proposed automated mirrors:** `sea_dbs_lowest_tail`; `gs_highest_or_near_highest_tail`; `pm_not_sea` (PM late mean closer to Baseline than to SEA).

---

## Conventions (v1)

| Knob | v1 choice |
|------|-----------|
| Seed | `0` (lock for one-seed gates) |
| RL step | **2 ms** × **30** steps/episode (biomarker window convention: see sea_dbs replication.md) |
| Training episodes | **150** |
| Reward | Eq. (7), $\beta_t = 0.35$ |
| Baseline variant | `baseline` — learning DDPG **without** PM/GS (not frozen) |
| Full SEA-DBS | `paper` — PM + GS |
| Inference carrier | Fixed **50 Hz** or **30 Hz** eval knob (not RL action) |
| PTQ | **FP16 PTQ only** (no QAT) |
| Gate redesign source | Digitized `paper.png` crops (this doc + `paper_digitization/curves.json`) |
