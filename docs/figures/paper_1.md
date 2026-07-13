# Mehregan et al. (paper 1) — figure comparisons

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/papers/`; JSON caches to `artifacts/figures/papers/`.

**Passed panels** (1b, 2a, 2b, 4a, 4b) use a short **Status** block. **Open panels** keep a full side-by-side checklist until gates pass.

| Panel | Script | Spec | Status |
|-------|--------|------|--------|
| Fig 1b — GPi PSD | `scripts/figures/papers/1/1b/plot.py` | [plant.md](../plant.md) | Pass |
| Fig 2a — GPi $P_\beta$ time series | `scripts/figures/papers/1/2a/plot.py` | [plant.md](../plant.md) | Pass |
| Fig 2b — Error Index time series | `scripts/figures/papers/1/2b/plot.py` | [plant.md](../plant.md) | Pass |
| Fig 4a — training $P_\beta$ vs step | `scripts/figures/papers/1/4a/plot.py` | [environment.md](../environment.md), [ddpg/replication.md](../controllers/ddpg/replication.md) | Pass (v4) |
| Fig 4b — training reward vs episode | `scripts/figures/papers/1/4b/plot.py` | [environment.md](../environment.md), [ddpg/replication.md](../controllers/ddpg/replication.md) | Pass |
| Fig 5a — post-train efficacy @ 45 Hz | `scripts/figures/papers/1/5a/plot.py` | [environment.md](../environment.md), [ddpg/replication.md](../controllers/ddpg/replication.md) | Open (interim plot) |
| Fig 5b — post-train efficacy @ 30 Hz | `scripts/figures/papers/1/5b/plot.py` | [environment.md](../environment.md), [ddpg/replication.md](../controllers/ddpg/replication.md) | Open (interim plot) |
| Fig 6a — PTQ / QAT @ 45 Hz | `scripts/figures/papers/1/6a/plot.py` | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) | Open |
| Fig 6b — PTQ / QAT @ 30 Hz | `scripts/figures/papers/1/6b/plot.py` (planned) | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) | Open |

Replication PNGs: `figures/papers/`. JSON caches: `artifacts/figures/papers/`.

---

## Fig 1b — GPi PSD

Mean GPi multitaper power spectral density (1–50 Hz) for three conditions: **healthy control**, **PD no treatment**, and **PD + 130 Hz STN cDBS**. Ordering gate: **PD > healthy** on beta power and **130 Hz cDBS < PD** (see [plant.md](../plant.md)).

### Paper (Mehregan et al.)

![Paper Fig 1b](papers/1/1b/paper.png)

### Replication

![Replication Fig 1b](papers/1/1b/gpi_psd.png)

<!-- caption-1b:start -->
**Caption:** see manifest

**Manifest:** `artifacts/figures/papers/1/1b/manifest.json`
<!-- caption-1b:end -->

**Status:** Pass — condition ordering and beta-peak shape match the paper panel (seeds `0–9` mean).

**Run:**

```bash
uv run python scripts/figures/papers/1/1b/plot.py
uv run python scripts/figures/papers/1/1b/plot.py --plot-only
```

**Defaults:** seeds `0–9` (mean PSD), 10 s segment, Python plant.

---

## Fig 2a — GPi $P_\beta$ time series

GPi beta-band power ($P_\beta$, Eq. 1, 13–35 Hz) over **12 s**: **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Shared baseline 0–2 s; dashed vertical at **2 s** (cDBS onset for blue). After onset, blue falls to a low plateau; red stays elevated.

### Paper (Mehregan et al.)

![Paper Fig 2a](papers/1/2a/paper.png)

### Replication

![Replication Fig 2a](papers/1/2a/beta_power.png)

<!-- caption-2a:start -->
**Caption:** 14 s sim (2 s pre-roll), plot = sim − 2 s, 0.2 s trailing / 2 s window (end sim 14 s), seed 0 (2026-07-11)

**Manifest:** `artifacts/figures/papers/1/2a/manifest.json`
<!-- caption-2a:end -->

**Status:** Pass — blue-below-red after $t=2$, shared 0–2 s baseline, dense trailing protocol. Protocol: trailing windows end at sim **14 s** (display $t=12$ → `[12, 14]`); enlarged Numba GPI spike buffer (904) so recording is not truncated. Remaining polish: blue floor slightly below paper at $t=12$; single seed (0).

**Run:**

```bash
uv run python scripts/figures/papers/1/2a/plot.py
uv run python scripts/figures/papers/1/2a/plot.py --plot-only
uv run python scripts/figures/papers/1/2a/plot.py --sampling segment
```

**Defaults:** seed `0`, 0.2 s trailing samples, 2 s overlapping window, 14 s integrate with 2 s pre-roll.

---

## Fig 2b — Error Index time series

Windowed Error Index (EI, Eq. 2) over **12 s** with **So-style SMC pulses into TH** (path A): BoC inverse-gamma on **Iappth**, `iappth_baseline=0`, `ggith=0.112`. **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Same timing as Fig 2a. Y-axis **Error Index** (replication default **0.10–0.4**; paper panel reads ~0–0.4).

### Paper (Mehregan et al.)

![Paper Fig 2b](papers/1/2b/paper.png)

### Replication

![Replication Fig 2b](papers/1/2b/error_index_v2.png)

<!-- caption-2b:start -->
**Caption:** 14 s sim (2 s pre-roll), plot = sim − 2 s, 0.2 s trailing / 2 s EI window (end sim 14 s), SMC BoC inv-gamma Iappth, backend python, seed 0, v2, y-axis 0.10–0.4 (2026-07-13)

**Manifest:** `artifacts/figures/papers/1/2b/manifest.json`
<!-- caption-2b:end -->

**Status:** Pass — blue-below-red after $t=2$, shared baseline, blue floor ~0.12 near paper. Remaining polish: red $t=12$ slightly low (~0.24 vs ~0.30); single seed.

**Run:**

```bash
uv run --group figures python scripts/figures/papers/1/2b/plot.py
uv run --group figures python scripts/figures/papers/1/2b/plot.py --plot-only
```

Each run writes a new ``figures/papers/1/2b/error_index_vN.png`` (N auto-increments) and updates the replication image link above.

**Defaults:** seed `0`, `smc_site='thalamic'`, `iappth_baseline=0`, `ggith=0.112`, `smc_amplitude=3.5`, `smc_schedule='boc'`, `smc_pulse_source='drive'`, backend **python**.

### Convention (path A, 2026-07-12)

**Citations (split roles):** Gao et al. (ICCPS 2020) define the **EI metric** Mehregan uses (exactly one TH spike in $(\mathrm{SMC}_\tau,\mathrm{SMC}_\tau{+}25\,\mathrm{ms})$; windowed $T_\omega{=}2\,\mathrm{s}$). So et al. (2012) define the **TH drive** for that metric (SMC current pulses into TH; TH not spontaneously active). Kumaravelu replaced those pulses with constant $I_{\mathrm{appth}}=1.2$; Fig 2b restores So-style drive: **pulses only** (`iappth_baseline=0`) plus BoC inverse-gamma timing (~14 Hz mean). Cortical `Iappco` SMC remains available but does **not** produce paper ordering (no Cor→TH synapse). Sweep: `artifacts/probes/fig2b_ei_so_path_a_sweep.json`.

---

## Fig 4a — training beta power vs step

Per-step GPi beta-band power during DDPG training of the **45 Hz** mean-frequency model (§IV.A.1): **300** environment steps (10 episodes × 30 steps). Y-axis **PSD(x10³)** = raw $P_\beta / 1000$ (same scale as the paper panel). The paper trace is noisy early (~0.43–0.57), then drops sharply around step **130–150** and settles lower (~0.35–0.45).

### Paper (Mehregan et al.)

![Paper Fig 4a](papers/1/4a/paper.png)

### Replication

![Replication Fig 4a](papers/1/4a/training_beta_v4.png)

<!-- caption-4a:start -->
**Caption:** 45 Hz fixed_mean_pattern, within_step L=1, reward=full_segment, softmax, critic=one_hot, seed 0, v4 (same training trace as v3; y-axis auto-fit so late points below 0.3 are visible), init_bias=0.5, early=0.428 late=0.299, trend↓ (2026-07-13). Locked as preferred replication — qualitative high→drop→low shape matches the paper panel.

**Manifest:** `artifacts/figures/papers/1/4a/manifest.json`
<!-- caption-4a:end -->

**Status:** Pass (v4) — same seed-0 training run as v3; plot y-limits extended to show the full trace. Qualitative shape match (noisy early, drop ~130–150, lower late). Late mean sits a bit below the paper’s ~0.35–0.45 band; accepted as polish, not a blocker.

### Side-by-side checklist

Qualitative gates first; numeric bands are approximate (paper read from panel; replication from `artifacts/figures/papers/1/4a/manifest.json`).

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Plot style** | Single noisy line, 0–300 steps | Line trace, 300 steps | ✓ |
| **Axes** | Steps 0–300; y **PSD(x10³)** ~0.3–0.6 | Same labels; y auto-fits full trace (v4: 0.2–0.6) | ✓ |
| **Early band (steps 0–130)** | ~0.43–0.57, high variance | mean 0.428 | ✓ |
| **Drop timing** | Sharp fall ~step 130–150 | mid(120–150) mean 0.403 | ✓ |
| **Late band (steps 150–300)** | ~0.35–0.45 | mean 0.299 (a bit low; shape OK) | ~ |
| **Overall trend** | Mean beta **decreases** over training | end−start window Δ=-0.133 | ✓ |

**Run:**

```bash
uv run python scripts/figures/papers/1/4a/plot.py
uv run python scripts/figures/papers/1/4a/plot.py --plot-only
```

Each run writes a new ``figures/papers/1/4a/training_beta_vN.png`` (N auto-increments) and updates the replication image link above. Prefer keeping the locked **v4** image unless intentionally re-promoting.

Long run (~30–60 min Python plant). Use tmux:

```bash
tmux new-session -d -s fig4a-train \
 "setsid nohup uv run python scripts/figures/papers/1/4a/plot.py >> logs/fig4a-train.log 2>&1 < /dev/null"
```

**Defaults:** seed `0`, **45 Hz** mean init, `state_length=1`, `fixed_mean_pattern`, **softmax** exploration (τ 3→1), **`critic_action_input=one_hot`**, `init_bias_scale=0.5`, `plant.dt_ms=0.02`.

---

## Fig 4b — training reward vs episode

Episode **total reward** and **episode-mean PSD(x10³)** during the same **45 Hz** DDPG run as Fig 4a (§IV.A.1). The paper panel indexes episodes **0–8** (line reaches episode 8; ticks every 2): reward rises from roughly **−80** toward **0** by episodes **4–6**, while episode-mean PSD falls inversely (~0.50 → ~0.37). We plot these as **two separate panels** (9 episodes, indices 0–8).

### Paper (Mehregan et al.)

![Paper Fig 4b](papers/1/4b/paper.png)

### Replication

**Reward vs episode**

![Replication Fig 4b reward](papers/1/4b/training_reward_v13.png)

**Episode-mean PSD vs episode**

![Replication Fig 4b PSD](papers/1/4b/training_psd_v13.png)

<!-- caption-4b:start -->
**Caption:** 9 episodes, 45 Hz fixed_mean_pattern (Fig 4a paired run), seed 0, source series_v4.json, v13, reward ep0=-29.1 ep8=16.1, rise_ep=3, psd 0.437→0.292, gate pass (2026-07-13)

**Manifest:** `artifacts/figures/papers/1/4b/manifest.json`
<!-- caption-4b:end -->

**Status:** Pass — two panels × 9 episodes (indices 0–8), paired with locked Fig 4a v4 (**seed 0**; paper seed unspecified). Qualitative match: reward↑, episode-mean PSD↓, rise by ~ep 3–5. Y-limits snap to data extrema (reward step 10, PSD step 0.05). Numeric bands differ from paper (reward scale, late level, PSD shape); compare **trends**, not pointwise values, across seeds.

**Seed note:** Mehregan et al. do not report the training RNG seed. Our replication locks **seed 0** (Fig 4a v4 cache). Same protocol, different seed → different wiggles and levels; do not expect paper-exact −80→0 reward on one seed alone.

### Side-by-side checklist

Qualitative gates first; numeric bands are approximate (paper read from panel; replication from `artifacts/figures/papers/1/4b/manifest.json`, seed 0).

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Plot style** | One panel; reward + PSD vs episode 0–8 | Two PNGs; episodes 0–8 (0-based), line reaches ep 8 | ✓ |
| **Axes (episodes)** | 0–8, ticks every 2 | 0–8, ticks every 2 | ✓ |
| **Axes (reward scale)** | ~−80–0 | auto snap −30–20 (data min/max) | ~ |
| **Axes (PSD scale)** | ~0.35–0.50 | auto snap 0.25–0.45 (data min/max) | ~ |
| **Early episodes (0–2)** | Reward negative (~−80 to ~−55) | mean −28.3 (negative, different magnitude) | ~ |
| **Rise timing** | Climb ~ep 2–6 toward ~0 | first rise by episode 3; jump ~ep 5 | ~ |
| **Late episodes (6–8)** | Plateau near **0** | mean +14.3 (improved, not near 0) | ~ |
| **Episode-mean PSD** | Gradual fall ~0.50→~0.37 | 0.437→0.292; cliff ~ep 5 vs paper wiggles | ~ |
| **Automation gate** | — | `ep_last > ep1` or recovery > 20 (recovery 45.2) | ✓ |

**Run:**

```bash
uv run python scripts/figures/papers/1/4b/plot.py
uv run python scripts/figures/papers/1/4b/plot.py --plot-only
```

Each run writes new ``training_reward_vN.png`` and ``training_psd_vN.png`` (same N) and updates the replication links above.

**Defaults:** **9 episodes** (indices **0–8**) from locked Fig 4a **v4** (`series_v4.json`, **seed 0**). Y-limits: ceil/floor of data extrema (reward tick step 10, PSD step 0.05). Locked replication images: **v13**.

---

## Fig 5a — post-train efficacy @ 45 Hz

Post-training evaluation on the **45 Hz** model (§IV.A.2): **12 s** display = **2 s** baseline (shared pre-stim) + **5** repeated **2 s** stimulation steps. Step-function **GPi** $P_\beta$ (raw PSD scale, 100–600 in the paper panel) for four conditions on the **same seed**:

1. **PD no stim** (black)
2. **Fully trained** 45 Hz pattern policy (green)
3. **Periodic 45 Hz** (pattern 0 / regular train init) (orange)
4. **Periodic 130 Hz** cDBS (yellow)

Dashed vertical at **2 s** (stimulation onset). Paper claims: trained stimulation **reduces** beta vs no stim after onset and shows efficacy at the **fixed 45 Hz** mean rate (not necessarily the lowest trace — 130 Hz cDBS is lower).

### Paper (Mehregan et al.)

![Paper Fig 5a](papers/1/5a/paper.png)

### Replication

![Replication Fig 5a](papers/1/5a/efficacy_45hz.png)

<!-- caption-5a:start -->
**Caption:** 45 Hz paper-protocol eval, Python plant, seed 0; post-onset means: no_stim=478, trained=300, periodic=300, cdbs130=199; automation gates pass (2026-07-13)

**Manifest:** `artifacts/figures/papers/1/5a/manifest.json`
<!-- caption-5a:end -->

**Status:** Open (interim plot) — four-series panel promoted; automation gates pass (shared baseline, trained &lt; no stim, 130 Hz lowest). **Remaining gap:** trained policy collapsed to **pattern 0** (green ≡ orange); paper expects trained **above** periodic 45 Hz on raw $P_\beta$.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 2 s baseline + 5×2 s steps; fixed seed | seed 0, 5×2 s steps | ✓ |
| **Series** | no stim, trained 45 Hz, periodic 45 Hz, 130 Hz cDBS | all four plotted | ✓ |
| **Shared baseline (0–2 s)** | Traces overlap pre-onset | Δ = 0 (~489) | ✓ |
| **Ordering after onset** | **130 Hz** lowest; trained **< no stim** | cdbs130=199; trained=300 &lt; no_stim=478 | ✓ |
| **Trained vs periodic 45 Hz** | Trained above periodic 45 Hz on raw $P_\beta$ | **Identical** (300 = 300; action 0 collapse) | ✗ |

**Run:**

```bash
# Full pipeline (train + eval + plot, ~30–60 min train):
uv run python scripts/figures/papers/1/5a/plot.py

# Eval + plot only (after checkpoint exists):
uv run python scripts/figures/papers/1/5a/plot.py --no-train --run-eval \
  --checkpoint artifacts/figures/papers/1/5a/checkpoint.pt

# Replot from cached eval JSON:
uv run python scripts/figures/papers/1/5a/plot.py --plot-only
```

Long run — use tmux:

```bash
tmux new-session -d -s fig5a-train \
 "setsid nohup uv run python scripts/figures/papers/1/5a/plot.py >> logs/fig5a-train.log 2>&1 < /dev/null"
```

**Defaults:** seed `0`, Python plant, `plant.dt_ms=0.02`, `state_length=1`, `fixed_mean_pattern`, `pattern_mean_hz=45`.

---

## Fig 5b — post-train efficacy @ 30 Hz

Same **12 s** paper-protocol eval for the **30 Hz** trained model (§IV.A.2). Three conditions:

1. **PD no stim** (black)
2. **Fully trained** 30 Hz pattern policy (green)
3. **Periodic 30 Hz** (pattern 0) (orange)

Key paper claim: **periodic 30 Hz elevates** beta (stimulation rate inside the beta band); the **trained irregular** pattern **lowers** beta below both no stim and periodic 30 Hz.

### Paper (Mehregan et al.)

![Paper Fig 5b](papers/1/5b/paper.png)

### Replication

![Replication Fig 5b](papers/1/5b/efficacy_30hz.png)

<!-- caption-5b:start -->
**Caption:** Interim 30 Hz paper-protocol eval (task108 JSON), PSD(x10³) scale, y-axis 0.10–0.70; periodic mean=0.627, trained mean=0.514 (seed 0).

**Manifest:** `artifacts/figures/papers/1/5b/manifest.json`
<!-- caption-5b:end -->

**Status:** Open — hardest computational gate; prior 30 Hz retrains **failed** trained `<` no-stim (see [replication-fidelity.md](../development/replication-fidelity.md)).

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 2 s baseline + 5×2 s steps; fixed seed | TBD | — |
| **Periodic 30 Hz vs no stim** | Periodic **>** no stim after $t=2$ | TBD | — |
| **Trained vs no stim** | Trained **<** no stim after $t=2$ | TBD | — |
| **Trained vs periodic 30 Hz** | Trained **<** periodic 30 Hz | TBD | — |
| **Qualitative shape** | Green low band ~350–430; orange high ~580–650 | TBD | — |

**Interim run:**

```bash
uv run python scripts/run_task108_paper_protocol_eval.py --mean-hz 30 \
  --landscape artifacts/ddpg/pattern_reward_landscape_30hz.json \
  --checkpoint artifacts/ddpg/<trained_30hz_checkpoint>.pt
uv run python scripts/figures/plot_beta_psd_paper_figures.py
```

Writes interim PNG to `artifacts/ddpg/fig5b_beta_psd_30hz.png`.

**Defaults:** seed `0`, `plant.dt_ms=0.02`, `pattern_mean_hz=30`; same env stack as 45 Hz except mean init.

**Acceptance (automation):** trained mean $P_\beta$ `<` no-stim **and** `<` periodic 30 Hz (`_fig5b_pass` in training scripts).

---

## Fig 6a — PTQ / QAT @ 45 Hz

Quantization comparison on the **45 Hz** trained policy (§IV.A.3): step-function $P_\beta$ over the same **12 s** eval protocol. Four series:

1. **Fully trained** (fp32) (green)
2. **PTQ, int8** (blue)
3. **PTQ, fp16** (purple)
4. **QAT** (dashed orange)

Paper claim: **PTQ** (fp16 and int8) tracks full-precision beta suppression after onset; **QAT** (10 episodes) **fails** to reduce beta and stays near the pre-stim level.

### Paper (Mehregan et al.)

![Paper Fig 6a](papers/1/6a/paper.png)

### Replication

*Not yet generated.* Target: `figures/papers/1/6a/ptq_qat_45hz.png`

<!-- caption-6a:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/1/6a/manifest.json` (planned)
<!-- caption-6a:end -->

**Status:** Open — plot script promoted; requires fresh fp32 + QAT training (no architecture-compatible legacy checkpoints). Qualitative gates: PTQ fp16/int8 track fp32 after onset; QAT stays elevated.

**Run:**

```bash
uv run python scripts/figures/papers/1/6a/plot.py
uv run python scripts/figures/papers/1/6a/plot.py --plot-only
uv run python scripts/figures/papers/1/6a/plot.py --skip-train \
  --fp32-checkpoint artifacts/figures/papers/1/6a/fp32_train0.pt \
  --qat-checkpoint artifacts/figures/papers/1/6a/qat_train0.pt
```

Long run (~1–2 h). Use tmux:

```bash
tmux new-session -d -s fig6a-train \
 "setsid nohup uv run python scripts/figures/papers/1/6a/plot.py >> logs/fig6a-train.log 2>&1 < /dev/null"
```

**Defaults:** seed `0`, Fig 4a DDPG profile (45 Hz, softmax + one_hot critic), 10-episode fp32 + QAT trains; eval protocol matches Fig 5a (2 s baseline + 5×2 s steps); raw PSD y-axis.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | Same 12 s eval as Fig 5a | TBD | — |
| **PTQ fp16 / int8 vs fp32** | Track closely after $t=2$ (~320–430 band) | TBD | — |
| **QAT vs fp32** | QAT stays **elevated** (~420–510); does not suppress beta | TBD | — |
| **Onset marker** | Dashed vertical at **2 s** | TBD | — |

**Interim run:**

```bash
# After fp32 checkpoint + paper-protocol eval JSON exist:
uv run python scripts/figures/plot_beta_psd_paper_figures.py \
  --fig6-json artifacts/ddpg/<eval_45hz_ptq_int8>.json \
  --fig6-json artifacts/ddpg/<eval_45hz_ptq_fp16>.json \
  --fig6-json artifacts/ddpg/<eval_45hz_qat>.json
```

Benchmark slugs: `ptq-fp16`, `ptq-int8`, `qat` ([benchmarking.md](../benchmarking.md)).

**Defaults:** 45 Hz trained actor; eval seed `0`; `plant.dt_ms=0.02`.

---

## Fig 6b — PTQ / QAT @ 30 Hz

Same quantization panel layout as Fig 6a for the **30 Hz** trained model (§IV.A.3): fp32, PTQ int8, PTQ fp16, and QAT. Paper shows the same qualitative split — PTQ tracks fp32 suppression; QAT remains high.

### Paper (Mehregan et al.)

![Paper Fig 6b](papers/1/6b/paper.png)

### Replication

*Not yet generated.* Target: `figures/papers/1/6b/ptq_qat_30hz.png`

<!-- caption-6b:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/1/6b/manifest.json` (planned)
<!-- caption-6b:end -->

**Status:** Open — blocked on passing 30 Hz trained policy (Fig 5b); QAT failure mode should reproduce even if fp32 is weak.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | Same 12 s eval as Fig 5b | TBD | — |
| **PTQ fp16 / int8 vs fp32** | Track closely after $t=2$ | TBD | — |
| **QAT vs fp32** | QAT elevated; no beta suppression | TBD | — |
| **Mean-rate context** | 30 Hz fp32 may sit higher than 45 Hz panel | TBD | — |

**Interim run:** same as Fig 6a with `--json-30hz` eval payloads and 30 Hz checkpoints.

**Defaults:** 30 Hz trained actor; eval seed `0`; `plant.dt_ms=0.02`.
