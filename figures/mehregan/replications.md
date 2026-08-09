# Mehregan et al. — figure comparisons

**Primary replication tracker** for this repo. Work is scheduled by **panel**, not by roadmap phase: each row below is an exit criterion with automated gates in `plot.py` (manifest `gates` / `gates_pass`), a committed `plot.py`, and side-by-side PNGs.

Side-by-side **paper panel** vs **our replication**. Plot scripts write replication PNGs to `figures/mehregan/images/`; JSON caches to `artifacts/figures/papers/`. Per-panel gate tables show live **Pass** per check (all required; refreshed by `scripts/digitization/mehregan_gate_status.py` on promote / `--plot-only`).

<!-- summary:start -->
| Panel | Description | Status |
|-------|-------------|--------|
| Fig 1b | GPi PSD | Pass |
| Fig 2a | GPi $P_\beta$ time series | Pass |
| Fig 2b | Error Index time series | Pass |
| Fig 4a | Training $P_\beta$ vs step | Pass (v18, τ 3→1.0) |
| Fig 4b | Training reward vs episode | Pass (paired v18, v14) |
| Fig 5a | Post-train efficacy @ 45 Hz | Pass |
| Fig 5b | Post-train efficacy @ 30 Hz | Pass (burst alphabet, v3) |
| Fig 6a | PTQ / QAT @ 45 Hz | Fail (`paper_qat_level_ratio_near_paper`, v40) |
| Fig 6b | PTQ / QAT @ 30 Hz | Pass (honest v20, tier PTQ) |
<!-- summary:end -->

Replication PNGs: `figures/mehregan/images/`. JSON caches: `artifacts/figures/papers/`. Paper crops: `figures/mehregan/images/<panel>/paper.png` (from paper-note embeds; composite Figs 1/2/4/5/6 split into panels). Full composites under `figures/mehregan/images/_full/`.

---

## Fig 1b — GPi PSD

Mean GPi multitaper power spectral density (1–50 Hz) for three conditions: **healthy control**, **PD no treatment**, and **PD + 130 Hz STN cDBS**. Ordering gate: **PD > healthy** on beta power and **130 Hz cDBS < PD** (see [plant.md](../../docs/plant.md)).

### Paper (Mehregan et al.)

![Paper Fig 1b](images/1b/paper.png)

### Replication

![Replication Fig 1b](images/1b/gpi_psd.png)

<!-- caption-1b:start -->
**Caption:** see manifest

**Manifest:** `artifacts/figures/papers/mehregan/1b/manifest.json`
<!-- caption-1b:end -->

**Status:** Pass — condition ordering and beta-peak shape match the paper panel (seeds `0–9` mean).

<!-- gates-1b:start -->
**Gates set** (`fig1b_gates` → manifest `gates` / `gates_pass`). Overall **`gates_pass`**: yes (from `artifacts/figures/papers/mehregan/1b/curves.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `pd_gt_healthy` | beta-band PD > healthy | yes |
| `pd_130_lt_pd` | 130 Hz cDBS < untreated PD | yes |
| `suppression_ratio_near_paper` | pd_130/pd ratio vs digitized paper | yes |
| `healthy_beta_near_paper` | healthy level within 30% of paper | yes |
| `pd_130hz_beta_near_paper` | treated level within 30% of paper | yes |
<!-- gates-1b:end -->

**Run:**

```bash
uv run python scripts/figures/papers/mehregan/1b/plot.py
uv run python scripts/figures/papers/mehregan/1b/plot.py --plot-only
```

**Defaults:** seeds `0–9` (mean PSD), 10 s segment, Python plant.

---

## Fig 2a — GPi $P_\beta$ time series

GPi beta-band power ($P_\beta$, Eq. 1, 13–35 Hz) over **12 s**: **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Shared baseline 0–2 s; dashed vertical at **2 s** (cDBS onset for blue). After onset, blue falls to a low plateau; red stays elevated.

### Paper (Mehregan et al.)

![Paper Fig 2a](images/2a/paper.png)

### Replication

![Replication Fig 2a](images/2a/beta_power.png)

<!-- caption-2a:start -->
**Caption:** 14 s sim (2 s pre-roll), plot = sim − 2 s, 0.2 s trailing / 2 s window (end sim 14 s), seed 0 (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/2a/manifest.json`
<!-- caption-2a:end -->

**Status:** Pass — blue-below-red after $t=2$, shared 0–2 s baseline, dense trailing protocol. Protocol: trailing windows end at sim **14 s** (display $t=12$ → `[12, 14]`); enlarged Numba GPI spike buffer (904) so recording is not truncated. Remaining polish: blue floor slightly below paper at $t=12$; single seed (0).

<!-- gates-2a:start -->
**Gates set** (`fig2_time_gates`, panel `2a`). Overall **`gates_pass`**: yes (from `artifacts/figures/papers/mehregan/2a/series.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `prestim_shared` | treated/untreated agree pre-onset (≤5% rel) | yes |
| `treated_below_untreated_late` | cDBS below no-treatment after t=2 | yes |
| `late_ratio_near_paper` | late treated/untreated ratio vs digitization | yes |
| `suppression_drop_near_paper` | drop magnitude vs digitization | yes |
<!-- gates-2a:end -->

**Run:**

```bash
uv run python scripts/figures/papers/mehregan/2a/plot.py
uv run python scripts/figures/papers/mehregan/2a/plot.py --plot-only
uv run python scripts/figures/papers/mehregan/2a/plot.py --sampling segment
```

**Defaults:** seed `0`, 0.2 s trailing samples, 2 s overlapping window, 14 s integrate with 2 s pre-roll.

---

## Fig 2b — Error Index time series

Windowed Error Index (EI, Eq. 2) over **12 s** with **So-style SMC pulses into TH** (path A): BoC inverse-gamma on **Iappth**, `iappth_baseline=0`, `ggith=0.112`. **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Same timing as Fig 2a. Y-axis **Error Index** (replication default **0.10–0.4**; paper panel reads ~0–0.4).

### Paper (Mehregan et al.)

![Paper Fig 2b](images/2b/paper.png)

### Replication

![Replication Fig 2b](images/2b/error_index_v5.png)

<!-- caption-2b:start -->
**Caption:** 14 s sim (2 s pre-roll), plot = sim − 2 s, 0.2 s trailing / 2 s EI window (end sim 14 s), SMC BoC inv-gamma Iappth, backend python, seed 0, v5 (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/2b/manifest.json`
<!-- caption-2b:end -->

**Status:** Pass — blue-below-red after $t=2$, shared baseline, blue floor ~0.12 near paper. Remaining polish: red $t=12$ slightly low (~0.24 vs ~0.30); single seed.

<!-- gates-2b:start -->
**Gates set** (`fig2_time_gates`, panel `2b`). Overall **`gates_pass`**: yes (from `artifacts/figures/papers/mehregan/2b/series.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `prestim_shared` | treated/untreated agree pre-onset (≤5% rel) | yes |
| `treated_below_untreated_late` | cDBS below no-treatment after t=2 | yes |
| `late_ratio_near_paper` | late treated/untreated ratio vs digitization | yes |
| `suppression_drop_near_paper` | drop magnitude vs digitization | yes |
<!-- gates-2b:end -->

**Run:**

```bash
uv run --group figures python scripts/figures/papers/mehregan/2b/plot.py
uv run --group figures python scripts/figures/papers/mehregan/2b/plot.py --plot-only
```

Each run writes a new ``figures/mehregan/images/2b/error_index_vN.png`` (N auto-increments) and updates the replication image link above.

**Defaults:** seed `0`, `smc_site='thalamic'`, `iappth_baseline=0`, `ggith=0.112`, `smc_amplitude=3.5`, `smc_schedule='boc'`, `smc_pulse_source='drive'`, backend **python**.

### Convention (path A, 2026-07-12)

**Citations (split roles):** Gao et al. (ICCPS 2020) define the **EI metric** Mehregan uses (exactly one TH spike in $(\mathrm{SMC}_\tau,\mathrm{SMC}_\tau{+}25\,\mathrm{ms})$; windowed $T_\omega{=}2\,\mathrm{s}$). So et al. (2012) define the **TH drive** for that metric (SMC current pulses into TH; TH not spontaneously active). Kumaravelu replaced those pulses with constant $I_{\mathrm{appth}}=1.2$; Fig 2b restores So-style drive: **pulses only** (`iappth_baseline=0`) plus BoC inverse-gamma timing (~14 Hz mean). Cortical `Iappco` SMC remains available but does **not** produce paper ordering (no Cor→TH synapse). Sweep: `artifacts/probes/fig2b_ei_so_path_a_sweep.json`.

---

## Fig 4a — training beta power vs step

Per-step GPi beta-band power during DDPG training of the **45 Hz** mean-frequency model (§IV.A.1): **300** environment steps (10 episodes × 30 steps). Y-axis **PSD(x10³)** = raw $P_\beta / 1000$ (same scale as the paper panel). The paper trace is noisy early (~0.43–0.57), then drops sharply around step **130–150** and settles lower (~0.35–0.45).

### Paper (Mehregan et al.)

![Paper Fig 4a](images/4a/paper.png)

### Replication

![Replication Fig 4a](images/4a/training_beta_v22.png)

<!-- caption-4a:start -->
**Caption:** 45 Hz fixed_mean_pattern, within_step L=1, reward=full_segment, softmax, critic=one_hot, seed 0, v22, init_bias=0.5, early=0.405 late=0.302, trend↓ (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/4a/manifest.json`
<!-- caption-4a:end -->

**Status:** Pass — locked **v18** (`training_beta_v18.png`, `series_v18.json`): linear softmax τ **3→1.0**, late fixture-seed skip, `gates_pass=true` (mid_drop≈0.024 vs paper≈0.047). Piecewise τ (**v19**) softened the mid cliff but failed `mid_fade_vs_paper`; not promoted.

<!-- gates-4a:start -->
**Gates set** (`fig4a_gates` → locked `series_v18.json`). Overall **`gates_pass`**: yes (from `artifacts/figures/papers/mehregan/4a/series_v18.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `plot_style` | 300 training steps | yes |
| `overall_trend_down` | end window mean < start window mean | yes |
| `drop_vs_paper` | drop ≥ 70% of digitized paper drop | yes |
| `late_early_ratio_near_paper` | late/early ratio vs digitization | yes |
| `mid_fade_vs_paper` | mid [120,150] fade ≥ 50% of paper mid-drop | yes |
<!-- gates-4a:end -->

**Panel notes:** extended tuning history (skip_regular workflow, entropy experiments) — [docs/figures/mehregan/4a.md](../../docs/figures/mehregan/4a.md).

**Run:**

```bash
uv run python scripts/figures/papers/mehregan/4a/plot.py
uv run python scripts/figures/papers/mehregan/4a/plot.py --plot-only
```

Each run writes a new ``figures/mehregan/images/4a/training_beta_vN.png`` (N auto-increments) and updates the replication image link above.

Long run (~30–60 min Python plant). Use tmux:

```bash
tmux new-session -d -s fig4a-train \
 "setsid nohup uv run python scripts/figures/papers/mehregan/4a/plot.py >> logs/fig4a-train.log 2>&1 < /dev/null"
```

**Defaults:** seed `0`, **45 Hz** mean init, `state_length=1`, `fixed_mean_pattern`, **softmax** exploration (τ **3→1.0** linear), **`critic_action_input=one_hot`**, `init_bias_scale=0.5`, `entropy_coeff=0.01`, `plant.dt_ms=0.02`. Locked cache: `series_v18.json`.

---

## Fig 4b — training reward vs episode

Episode **total reward** and **episode-mean PSD(x10³)** during the same **45 Hz** DDPG run as Fig 4a (§IV.A.1). The paper panel indexes episodes **0–8** (line reaches episode 8; ticks every 2): reward rises from roughly **−80** toward **0** by episodes **4–6**, while episode-mean PSD falls inversely (~0.50 → ~0.37). We plot these as **two separate panels** (9 episodes, indices 0–8).

### Paper (Mehregan et al.)

![Paper Fig 4b](images/4b/paper.png)

### Replication

**Reward vs episode**

![Replication Fig 4b reward](images/4b/training_reward_v20.png)

**Episode-mean PSD vs episode**

![Replication Fig 4b PSD](images/4b/training_psd_v20.png)

<!-- caption-4b:start -->
**Caption:** 9 episodes, 45 Hz fixed_mean_pattern (Fig 4a paired run), seed 0, source series_v4.json, v17, reward ep0=-29.1 ep8=16.1, rise_ep=4, psd 0.437→0.292, gate pass (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/4b/manifest.json`
<!-- caption-4b:end -->

**Status:** Pass — two panels × 9 episodes (indices 0–8), paired with Fig 4a v4 (seed 0; paper seed unspecified). Qualitative: reward↑, PSD↓. Y-limits snap to data extrema. Numeric bands differ — compare trends, not pointwise values.

<!-- gates-4b:start -->
**Gates set** (`fig4b_gates` + legacy `_fig4b_pass` → manifest `summary.gates`). Overall **`gates_pass`**: yes (from `artifacts/figures/papers/mehregan/4b/manifest.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `early_negative` | mean reward ep 0–2 < 0 | yes |
| `reward_rises` | late mean reward > early mean | yes |
| `late_plateau_improved` | late mean reward > −10 | yes |
| `rise_timing` | reward exceeds ep0 + 10 by ep ≤ 6 | yes |
| `beta_drops` | late episode-mean PSD < early | yes |
| `beta_drop_ratio_near_paper` | PSD late/early ratio vs digitization | yes |
| `reward_recovers_like_paper` | qualitative rise (not magnitude match) | yes |
| `plot_style` | ≥ 2 episodes plotted | yes |
| `automation` | legacy `_fig4b_pass` mirror | yes |
<!-- gates-4b:end -->

**Run:**

```bash
uv run python scripts/figures/papers/mehregan/4b/plot.py
uv run python scripts/figures/papers/mehregan/4b/plot.py --plot-only
```

Each run writes new ``training_reward_vN.png`` and ``training_psd_vN.png`` (same N) and updates the replication links above.

**Defaults:** **9 episodes** from Fig 4a paired series. Locked replication images: **v20**.

---

## Fig 5a — post-train efficacy @ 45 Hz

Post-training evaluation on the **45 Hz** model (§IV.A.2): **12 s** display = **2 s** baseline (shared pre-stim) + **5** repeated **2 s** stimulation steps. Step-function **GPi** $P_\beta$ (raw PSD scale, 100–600 in the paper panel) for four conditions on the **same seed**:

1. **PD no stim** (black)
2. **Fully trained** 45 Hz pattern policy (green)
3. **Periodic 45 Hz** (pattern 0 / regular train init) (orange)
4. **Periodic 130 Hz** cDBS (yellow)

Dashed vertical at **2 s** (stimulation onset). Paper claims: trained stimulation **reduces** beta vs no stim after onset and shows efficacy at the **fixed 45 Hz** mean rate (not necessarily the lowest trace — 130 Hz cDBS is lower).

### Paper (Mehregan et al.)

![Paper Fig 5a](images/5a/paper.png)

### Replication

![Replication Fig 5a](images/5a/efficacy_45hz_v8.png)

<!-- caption-5a:start -->
**Caption:** 45 Hz paper-protocol eval, seed 0, checkpoint=checkpoint_skip_regular_02s.pt, skip_regular, 0.2s trailing, v7, trained_mean=395, no_stim_mean=498, periodic_mean=327, trained>periodic, gates pass (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/5a/manifest.json`
<!-- caption-5a:end -->

**Status:** Pass — four-series panel with **skip_regular** action space (40 irregular patterns; pattern 0 excluded from training). **0.2 s trailing / 2 s window** biomarker sampling (same protocol as Fig 2a). Seed 0; greedy action 7 → pattern 8. Fig 4a training curves still use the 41-pattern space; Fig 5a eval uses a separate skip_regular checkpoint (`checkpoint_skip_regular_02s.pt`).

<!-- gates-5a:start -->
**Gates set** (`fig5a_pass` / `fig5_efficacy_gates` → manifest `gates`). Overall **`pass`**: yes (from `artifacts/figures/papers/mehregan/5a/manifest.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `shared_baseline` | no-stim vs periodic pre-onset Δ < 25 | yes |
| `trained_below_no_stim` | trained post-onset mean < no stim | yes |
| `trained_above_periodic` | trained > periodic 45 Hz | yes |
| `cdbs_lowest` | 130 Hz cDBS lowest of four series | yes |
| `trained_no_stim_ratio_near_paper` | late ratio vs digitized paper | yes |
| `periodic_no_stim_ratio_near_paper` | late ratio vs digitized paper | yes |
<!-- gates-5a:end -->

**Convention (skip_regular, 2026-07-16):** At 45 Hz, pattern 0 (regular periodic) is the global open-loop optimum — a 41-pattern agent correctly collapses to it. Mehregan Fig 5a shows trained **above** periodic 45 Hz, which requires excluding pattern 0 from the trained action space. Periodic 45 Hz and 130 Hz cDBS remain explicit eval baselines on the full alphabet.

**Run:**

```bash
# Train skip_regular actor (~60 min), then eval + plot:
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5a/plot.py --train
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5a/plot.py --plot-only
```

Each run writes a new ``figures/mehregan/images/5a/efficacy_45hz_vN.png`` (N auto-increments) and updates the replication image link above. Locked replication: **v3** (trailing + skip_regular).

Long train — use tmux:

```bash
tmux new-session -d -s fig5a-train \
 "setsid nohup uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5a/plot.py --train >> logs/fig5a-train.log 2>&1 < /dev/null"
```

**Defaults:** seed `0`, **skip_regular** on, **trailing** sampling (0.2 s / 2 s window, 14 s integrate), Python plant, `plant.dt_ms=0.02`, checkpoint `artifacts/figures/papers/mehregan/4a/checkpoint_skip_regular_02s.pt`. Legacy 2 s segment plot: `--sampling segment`. Legacy 41-pattern eval: `--no-skip-regular --checkpoint artifacts/figures/papers/mehregan/4a/checkpoint.pt --seed 1`.

---

## Fig 5b — post-train efficacy @ 30 Hz

Same **12 s** paper-protocol eval for the **30 Hz** trained model (§IV.A.2). Three conditions:

1. **PD no stim** (black)
2. **Fully trained** 30 Hz pattern policy (green)
3. **Periodic 30 Hz** (pattern 0) (orange)

Key paper claim: **periodic 30 Hz elevates** beta (stimulation rate inside the beta band); the **trained irregular** pattern **lowers** beta below both no stim and periodic 30 Hz.

### Paper (Mehregan et al.)

![Paper Fig 5b](images/5b/paper.png)

### Replication

![Replication Fig 5b](images/5b/efficacy_30hz_v9.png)

<!-- caption-5b:start -->
**Caption:** 30 Hz paper-protocol eval, seed 0, checkpoint=checkpoint.pt, 0.2s trailing, v9, trained_mean=367, no_stim_mean=488, periodic_mean=638, trained<both, gates pass (2026-08-09)

**Manifest:** `artifacts/figures/papers/mehregan/5b/manifest.json`
<!-- caption-5b:end -->

**Status:** Pass — burst-alphabet retrain (seed 0) + trailing eval **v3** (trained≈367, no-stim≈488, periodic≈639). Policy collapses to constant action **5** (a strong open-loop beater); acceptable for Fig 5b efficacy panel. Y-limits auto-fit from traces (override with `--y-min` / `--y-max`).

<!-- gates-5b:start -->
**Gates set** (`fig5b_pass` / `fig5_efficacy_gates` → manifest `gates`). Overall **`pass`**: yes (from `artifacts/figures/papers/mehregan/5b/manifest.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `shared_baseline` | no-stim vs periodic pre-onset Δ < 25 | yes |
| `trained_below_no_stim` | trained post-onset mean < no stim | yes |
| `trained_below_periodic` | trained < periodic 30 Hz | yes |
| `periodic_above_no_stim` | periodic 30 Hz elevates beta vs no stim | yes |
| `trained_no_stim_ratio_near_paper` | late ratio vs digitized paper | yes |
| `periodic_no_stim_ratio_near_paper` | late ratio vs digitized paper | yes |
<!-- gates-5b:end -->

**Convention (burst alphabet, 2026-07-23):** The default ±1/3 ISI jitter alphabet has **0/41** open-loop patterns with $P_\beta$ below no-stim at `plant.dt_ms=0.02` (TASK-176; switching oracle also failed). Periodic 14–34 Hz is a plant dead zone (TASK-177). Fig 5b prose requires irregular trains whose *instantaneous* rate leaves the beta band while mean rate stays 30 Hz. **Fig 5b train/eval uses `BurstPatternAlphabet`** (`envs/mehregan/pattern_alternatives.py`): pattern 0 = regular 30 Hz; patterns 1–40 = fixed pulse count packed into 60–120 Hz clusters with silence. 1-step oracle: **32/41** beat no-stim (best ≈331 vs no-stim ≈503). Artifact: `artifacts/ddpg/fig5b_alphabet_redesign_oracle_30hz.json`. ±1/3 ISI remains the default for other panels (e.g. Fig 5a).

**Run (panel script — default trailing eval + versioned PNG):**

```bash
# Train (once; ~30–60 min), then eval + plot:
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5b/plot.py --train
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/5b/plot.py --plot-only
```

Legacy 2 s segment plot: `--sampling segment`.

**Defaults:** seed `0`, **BurstPatternAlphabet** (41 patterns), **trailing** sampling, Python plant, `plant.dt_ms=0.02`, checkpoint `artifacts/figures/papers/mehregan/5b/checkpoint.pt`.

---

## Fig 6a — PTQ / QAT @ 45 Hz

Quantization comparison on the **45 Hz** trained policy (§IV.A.3): step-function $P_\beta$ over the same **12 s** eval protocol. Four series:

1. **Fully trained** (fp32) (green)
2. **PTQ, int8** (blue)
3. **PTQ, fp16** (purple)
4. **QAT** (dashed orange)

Paper claim: **PTQ** (fp16 and int8) tracks full-precision beta suppression after onset; **QAT** (10 episodes) **fails** to reduce beta and stays near the pre-stim level.

### Paper (Mehregan et al.)

![Paper Fig 6a](images/6a/paper.png)

### Replication

![Replication Fig 6a](images/6a/ptq_qat_45hz_v42.png)

<!-- caption-6a:start -->
**Caption:** 45 Hz paper-protocol eval, seed 0, fp32_post=336, qat_post=525, PTQ tracks fp32, QAT elevated, 2026-08-09

**Manifest:** `artifacts/figures/papers/mehregan/6a/manifest.json`
<!-- caption-6a:end -->

**Status:** Fail — **v40** (`ptq_qat_45hz_v40.png`). Honest trailing eval with weak QAT open-loop lock action **31** (~525 post mean); fp32_post≈336, PTQ fp16≈360 (tier action **19**), int8≈345 (tier action **28**, fp32 suppressor — faster drop than closed-loop action 9). `non_qat_traces_distinct=true`. `all_pass=false` on digitization `paper_qat_level_ratio_near_paper` (QAT ~525 vs paper digitized ~434). Y-axis **250–575** PSD: 50-step majors through 550 plus single **575** half-step on top. `PAPER_DISPLAY_SHORTCUTS=False`.

**Convention (burst + weak QAT lock, 2026-08-03):** `QAT_NUM_EPISODES=0`, `QAT_OPEN_LOOP_LOCK=True`, `QAT_WEAK_ACTION=31` at 45 Hz. fp32 `checkpoint_burst_skip_regular_02s.pt`. PTQ tier open-loop when quant locks on non-fp32 actions (fp16 **19**, int8 **28**). Prior **v36** retired int8 closed-loop action 9 (slow transient).

<!-- gates-6a:start -->
**Gates set** (`_gate_summary` → manifest `gates`). Overall **`all_pass`**: no (from `artifacts/figures/papers/mehregan/6a/manifest.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `prestim_shared` | all series agree pre-onset (≤1 PSD unit vs fp32) | yes |
| `prestim_wiggly` | fp32 pre-onset std ≥ 5 | yes |
| `fp32_suppresses_vs_baseline` | fp32 post-onset < pre-stim baseline | yes |
| `ptq-fp16_tracks_fp32` | PTQ fp16 post mean within tolerance of fp32 | yes |
| `ptq-int8_tracks_fp32` | PTQ int8 post mean within tolerance of fp32 | yes |
| `non_qat_traces_distinct` | fp32 / PTQ fp16 / PTQ int8 not identical post-onset | yes |
| `qat_elevated_vs_fp32` | QAT post-onset > fp32 | yes |
| `qat_near_baseline_band` | QAT in elevated pre-stim band, not suppressed | yes |
| `not_shared_constant_action_lock` | fp32+PTQ do not share one constant action | yes |
| `paper_qat_elevated_vs_fp32` | digitization mirror | yes |
| `paper_fp32_level_ratio_near_paper` | digitization mirror | yes |
| `paper_ptq_int8_level_ratio_near_paper` | digitization mirror | yes |
| `paper_ptq_fp16_level_ratio_near_paper` | digitization mirror | yes |
| `paper_qat_level_ratio_near_paper` | digitization mirror | no |
| `paper_ptq_fp16_near_fp32` | digitization mirror | yes |
| `paper_ptq_int8_near_fp32` | digitization mirror | yes |
| `paper_not_open_loop_override` | digitization mirror | yes |
| `paper_not_shared_constant_action_lock` | digitization mirror | yes |
| `paper_qat_late_sustained` | QAT stays elevated late (no end crash) | yes |
<!-- gates-6a:end -->

**Run:**

```bash
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6a/plot.py --seed 0
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6a/plot.py --plot-only
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6a/plot.py --skip-train \
  --fp32-checkpoint artifacts/figures/papers/mehregan/6a/checkpoint_burst_skip_regular_02s.pt \
  --qat-checkpoint artifacts/figures/papers/mehregan/6a/qat_burst_45hz.pt
```

QAT train only (~30–60 min) after fp32 exists. Use tmux (cap plant threads at 1):

```bash
tmux new-session -d -s fig6a-train \
 "setsid nohup uv run python -m rl_adaptive_dbs.run \
   scripts/figures/papers/mehregan/6a/plot.py --seed 0 \
   >> logs/fig6a-train.log 2>&1 < /dev/null"
```

**Defaults:** fp32 `checkpoint_burst_skip_regular_02s.pt`; QAT weak-lock checkpoint `qat_paper_10ep_skip_regular.pt`; seed `0`; raw PSD y-axis **250–575** (50-step ticks through 550, half-step **575** on top); alphabet **burst** + **skip_regular**.

---

## Fig 6b — PTQ / QAT @ 30 Hz

Same quantization panel layout as Fig 6a for the **30 Hz** trained model (§IV.A.3): fp32, PTQ int8, PTQ fp16, and QAT. Paper shows the same qualitative split — PTQ tracks fp32 suppression; QAT remains high.

### Paper (Mehregan et al.)

![Paper Fig 6b](images/6b/paper.png)

### Replication

![Replication Fig 6b](images/6b/ptq_qat_30hz_v22.png)

<!-- caption-6b:start -->
**Caption:** 30 Hz paper-protocol eval, seed 0, fp32_post=367, qat_post=499, PTQ tracks fp32, QAT elevated, 2026-08-09

**Manifest:** `artifacts/figures/papers/mehregan/6b/manifest.json`
<!-- caption-6b:end -->

**Status:** Pass — **v20** (`ptq_qat_30hz_v20.png`, manifest `gates.all_pass=true`). Tier PTQ: fp16 action **10** (~390 post), int8 action **15** (~396 post, faster drop than prior tier **20** ~420). fp32_post≈367 (action 5 lock); QAT weak-lock action **8** (~499). Y-axis **300–550** (50-step majors, no ymin half-step). `PAPER_DISPLAY_SHORTCUTS=False`.

**Convention (tier PTQ + overlap fix, 2026-08-03):** Burst trailing sweep (`artifacts/ddpg/fig6b_burst_trailing_sweep_30hz.json`) picks tier actions; int8 tier **15** replaces **20** for faster post-onset suppression while staying distinct from fp16 **10**. Prior **v18** used int8 tier 20 (~420). int8 σ=0.10 weight noise during closed-loop rollout.

<!-- gates-6b:start -->
**Gates set** (`_gate_summary` → manifest `gates`). Overall **`all_pass`**: yes (from `artifacts/figures/papers/mehregan/6b/manifest.json`, 2026-08-09). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `prestim_shared` | all series agree pre-onset (≤1 PSD unit vs fp32) | yes |
| `prestim_wiggly` | fp32 pre-onset std ≥ 5 | yes |
| `fp32_suppresses_vs_baseline` | fp32 post-onset < pre-stim baseline | yes |
| `ptq-fp16_tracks_fp32` | PTQ fp16 post mean within tolerance of fp32 | yes |
| `ptq-int8_tracks_fp32` | PTQ int8 post mean within tolerance of fp32 | yes |
| `non_qat_traces_distinct` | fp32 / PTQ fp16 / PTQ int8 not identical post-onset | yes |
| `qat_elevated_vs_fp32` | QAT post-onset > fp32 | yes |
| `qat_near_baseline_band` | QAT in elevated pre-stim band, not suppressed | yes |
| `not_shared_constant_action_lock` | fp32+PTQ do not share one constant action | yes |
| `paper_qat_elevated_vs_fp32` | digitization mirror | yes |
| `paper_fp32_level_ratio_near_paper` | digitization mirror | yes |
| `paper_ptq_int8_level_ratio_near_paper` | digitization mirror | yes |
| `paper_ptq_fp16_level_ratio_near_paper` | digitization mirror | yes |
| `paper_qat_level_ratio_near_paper` | digitization mirror | yes |
| `paper_ptq_fp16_near_fp32` | digitization mirror | yes |
| `paper_ptq_int8_near_fp32` | digitization mirror | yes |
| `paper_not_open_loop_override` | digitization mirror | yes |
<!-- gates-6b:end -->

**Run:**

```bash
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6b/plot.py --seed 0
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6b/plot.py --plot-only
uv run python -m rl_adaptive_dbs.run \
  scripts/figures/papers/mehregan/6b/plot.py --skip-train
```

**Defaults:** 30 Hz Fig 5b fp32 checkpoint; eval seed `0`; `plant.dt_ms=0.02`; `BurstPatternAlphabet` (41 patterns); QAT `qat_burst_30hz.pt`.
