# Ravivarapu et al. — figure replication gates

**Primary gate spec** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Ship status, side-by-side PNGs, and run commands live in [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md).

Controller / adapter: [sea_dbs/replication.md](../../controllers/sea_dbs/replication.md). Shared plant: [plant.md](../../plant.md). Schematics (paper Figs 1–3) are **out of scope**. Fig 2 (reward curve of Eq. (7)) is optional polish, not a gate.

**Defaults (paper §V.A / Table I):** seed `0` (paper seed unspecified — lock one seed for gates), **2 ms** RL step × **30** steps/episode × **150** training episodes, $\gamma=0.99$, buffer **8192**, batch **32**, $\alpha_a=5\times10^{-4}$, $\alpha_c=10^{-3}$. Variants: `baseline` (DDPG, no PM/GS) vs `paper` (full SEA-DBS).

| Panel | Script | Status |
|-------|--------|--------|
| Fig 4a — training PSD vs episode | `scripts/figures/papers/ravivarapu/4a/plot.py` | **Fail, gap documented** — lab notes [4a.md](../ravivarapu/4a.md) |
| Fig 4b — training reward vs episode | `scripts/figures/papers/ravivarapu/4b/plot.py` | **Fail, gap documented** — paired with 4a |
| Fig 5a — inference @ 50 Hz | `scripts/figures/papers/ravivarapu/5a/plot.py` | **Fail, gap documented** — [5a.md](../ravivarapu/5a.md) |
| Fig 5b — inference @ 30 Hz | `scripts/figures/papers/ravivarapu/5b/plot.py` | **Fail, gap documented** — [5b.md](../ravivarapu/5b.md) |
| Fig 6 — FP16 PTQ @ 50 Hz | `scripts/figures/papers/ravivarapu/6/plot.py` | **Fail, gap documented** — [6.md](../ravivarapu/6.md) |
| Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS) | `scripts/figures/papers/ravivarapu/7/plot.py` | **Fail, gap documented** — [7.md](../ravivarapu/7.md) |

Status as of the honest-replication pass (oct 12 2026): every shortcut that produced the earlier passes is removed ([sea_dbs/replication.md](../../controllers/sea_dbs/replication.md) §15). Live gates: [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md).

**Digitization:** refined WebPlotDigitizer curves per panel (`curves_fig{4a,4b,5a,5b,6,7}.json`); gates read them through `scripts/digitization/ravivarapu_gates.py`. Absolute y values have ~±0.01–0.02 (Fig 4a) / ~±10–15 (Figs 5–7) calibration uncertainty; **ordering, shared starts, relative drops, and cross-panel ratios** are the hard targets.

Digitization cache (local): `artifacts/figures/papers/ravivarapu/paper_digitization/curves_*.json`.

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

**Automated gates (2-tier suite, `ravivarapu_fig4a_gates`):** windows are early ep 0–15, mid 40–80, mid-late 70–110, late 120–150; "paper" values come from `curves_fig4a.json` over the same windows.

| Tier | Key | Rule (paper-faithful) |
|------|-----|------------------------|
| **Shape** | `n_episodes_ok` | $N \ge 150$ episodes for both variants |
| **Shape** | `shared_start` | $\|b_{\mathrm{early}} - s_{\mathrm{early}}\| \le 0.05$ |
| **Shape** | `baseline_declines` | $b_{\mathrm{early}} - b_{\mathrm{late}} \ge 0.02$ |
| **Shape** | `sea_below_baseline_mid` / `_midlate` / `_late` | SEA window mean below Baseline in each window |
| **Shape** | `sea_steeper_drop_than_baseline` | SEA early→late drop larger than Baseline's |
| **Shape** | `gap_widens_over_training` | Baseline − SEA gap does not close from mid → mid-late → late (−0.005 / −0.008 slack) |
| **Shape** | `drop_timing_baseline` / `_sea` | share of the drop reached by ep 50 ≤ max(1.55 × paper, paper + 0.10) |
| **Shape** | `gradual_decline_baseline` / `_sea` | mid − late ≥ 30 % of paper's mid − late |
| **Shape** | `pearson_baseline_min` / `_sea_min` | Pearson $r \ge 0.58$ vs the digitized curve |
| **Full** | `shared_start_near_paper` | both ep 0–15 means within 0.02 of paper |
| **Full** | `baseline_drop_vs_paper` / `sea_drop_vs_paper` | drop ≥ 40 % of paper's |
| **Full** | `baseline_mid_near_paper` / `_midlate_` / `_late_` | Baseline window means within 12 % of paper |
| **Full** | `sea_mid_near_paper` / `_midlate_` / `_late_` | SEA window means within 12 % of paper |
| **Full** | `mid_gap_near_paper` | mid gap in $[0.40, 1.80] \times$ paper |
| **Full** | `midlate_gap_near_paper` | mid-late gap in $[0.40, 1.75] \times$ paper |
| **Full** | `late_gap_near_paper` | late gap in $[0.50, 1.60] \times$ paper |
| **Full** | `late_early_ratio_baseline_near_paper` / `_sea_` | late/early ratio within 8 % of paper's |

28 gates, as set by the oct 14 2026 gate audit ([4a.md](../ravivarapu/4a.md) § Gate audit), which dropped `sea_declines` and `late_gap_substantial` as duplicates and tightened the start and late/early-ratio tolerances. The live table with pass/fail is in [replications.md](../../../figures/ravivarapu/replications.md) § Fig 4a.

Do **not** pass a frozen / non-learning Baseline. Do **not** use display hacks to force these levels.

**Stimulation drive convention:** every panel uses one plant setup — 150 ms independent plant shots per RL step, 50 Hz carrier on for the whole shot when $a = 1$, `observation_scale = 472` (Figs 5–7 override only the paper's stimulation frequency). The earlier per-variant 58 / 62 ms bursts at 130 Hz were tuned to the curves and are removed. Rationale and probe numbers: [sea_dbs/replication.md](../../controllers/sea_dbs/replication.md) §5, §15.

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

**Automated gates:** 20 keys (ordering, rise, timing, Pearson, absolute start and rise levels, and `eq7_consistent_with_4a`); the live table is in [replications.md](../../../figures/ravivarapu/replications.md) § Fig 4b, thresholds from the oct 14 2026 audit ([4a.md](../ravivarapu/4a.md) § Gate audit).

**Units (oct 12 2026):** both axes are the mean Eq. (7) reward per step, so `ravivarapu_fig4b_gates` compares levels, rises, and gaps to the digitized paper directly. It used to rescale every paper value by $|b_{\mathrm{early}} / p^{B}_{\mathrm{early}}|$ (our Baseline's early mean over the paper's), which let a run that started far above the paper's −1.3 pass the start and rise gates; `shared_start_near_paper` now checks both variants against the paper.

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
| 7 | **Steps 0–5** | Early window near the digitized paper | Early MAE above 20 % of the paper's 0→10 drop |

**Automated gates:** 20 keys — 11 PSD samples, shared start, both decline, SEA-DBS below and steeper than the Baseline, step 0→10 drops in $[0.5, 1.5] 	imes$ paper (0.104 / 0.148), a tail-gap band, early / mid MAE capped at a share of each paper series' drop, carrier. Live table: [replications.md](../../../figures/ravivarapu/replications.md) § Fig 5a. The eval uses the training plant setup (150 ms full-window 50 Hz shots).

---

## Fig 5b — inference @ 30 Hz

Same inference layout at **30 Hz** carrier (overlaps pathological beta; Fig. 5(b)).

**Qualitative gates:** weaker suppression than 5a; Baseline shows initial beta resonant rise/elevation at steps 1–2 before gradual decline; SEA-DBS shows a delayed, gentle drop (plateau) over steps 0–2 before smooth decline; SEA-DBS strictly below Baseline on late steps; both decline overall.

**Automated tiered gates (paper-anchored):**
- `shared_start`: Baseline and SEA-DBS start in high untreated band ($|s_0 - b_0| \le 0.05$).
- `early_baseline_rises`: Baseline rises or stays elevated at steps 1–2 ($b_1 \ge b_0 - 0.002$).
- `early_sea_plateau`: SEA-DBS delayed drop / plateau on steps 0–2 ($s_0 - s_1 \le 0.015$, $s_0 - s_2 \le 0.025$).
- `early_sea_below_baseline`: SEA-DBS below Baseline across early steps 1–5.
- `early_mae_*`: Steps 0–5 MAE vs the digitized curves capped at 20 % of each paper series' step 0→10 drop.
- `*_drop_near_paper`: step 0→10 drops in $[0.5, 1.5] \times$ paper (0.062 Baseline / 0.072 SEA-DBS); `tail_gap_near_paper`: SEA-below-Baseline tail gap band (a 0.0005 split fails).
- `late_*_declines`: Both traces keep declining through steps 5–10.
- `paper_end_below_baseline`: SEA-DBS ends below Baseline ($s_{10} < b_{10}$).
- `paper_steeper_drop`: Overall SEA-DBS drop exceeds Baseline drop.
- `pearson_*_min`: Pearson correlation $r \ge 0.70$ vs digitized paper trajectories.
- `weaker_than_50hz_*`: each 30 Hz variant has a positive drop smaller than its 50 Hz drop (a rising 30 Hz trace fails).

Thresholds as set by the oct 14 2026 audit ([4a.md](../ravivarapu/4a.md) § Gate audit); the early-ordering checks allow a 0.006 crossing because the paper's own SEA-DBS sits 0.003 above its Baseline at step 4. Live table: [replications.md](../../../figures/ravivarapu/replications.md) § Fig 5b.

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

**Automated gates:** 15 keys — four series with 11 samples, shared start, SEA-DBS (fp32 and PTQ) below the Baseline, PTQ tracking fp32 for both variants as a two-sided band (tails ≤ 0.03, any step ≤ 0.04), all four tails within 0.02 of the paper, early MAE, and `fp32_matches_fig5a` (the paper plots one rollout in Figs 5a / 6 / 7). Live table: [replications.md](../../../figures/ravivarapu/replications.md) § Fig 6.

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

**Automated gates:** 17 keys — `sea_dbs_lowest_tail`, `gs_highest_tail`, `pm_not_sea`, `baseline_above_pm_early`, tails within 0.02 of the paper for each variant, full-trace MAE for +PM / +GS, early MAE, and `fp32_matches_fig5a`. Live table: [replications.md](../../../figures/ravivarapu/replications.md) § Fig 7.

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
| Plant integration step | **0.02 ms** (§V.A; 0.01 ms before oct 17 2026, same levels) |
| Gate source | Refined WebPlotDigitizer curves `paper_digitization/curves_fig*.json` |
