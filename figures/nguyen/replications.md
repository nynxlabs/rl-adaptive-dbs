# Nguyen et al. — figure comparisons

**Primary replication tracker** for the DSQN / closed-loop neuromorphic DBS paper. Spec: [controllers/snn/replication.md](../../docs/controllers/snn/replication.md). Work proceeds **in parallel** with Mehregan ([Mehregan replications](../mehregan/replications.md)): each row is an exit criterion with automated gates, a committed `plot.py` (when present), and side-by-side PNGs.

Plot scripts write replication PNGs to `figures/nguyen/images/`; JSON caches to `artifacts/figures/papers/nguyen/`. Paper panels: `figures/nguyen/images/<panel>/paper.png`.

Figs **1–2** are schematics — **not** replication targets.

**Gates:** manifest `gates` in `artifacts/figures/papers/nguyen/<panel>/manifest.json`. Per-panel tables below refresh from manifest on each `promote` run (`scripts/figures/papers/promote.py` `refresh_nguyen_gate_tables`). Every listed check is required for exit. Digitization mirrors from `scripts/digitization/nguyen_gates.py` (`attach_digitization`).

<!-- summary:start -->
| Panel | Description | Status |
|-------|-------------|--------|
| Fig 3 | GPi α–β distribution (PD Off vs PD On) | Pass (rep v22) |
| Fig 4 | Training reward + episode length | Pass (rep v163) |
| Fig 5 | CBGT spikes + DBS energy over training | Pass (rep v62) |
| Fig 6 | α–β + DBS parameters over training | Pass (rep v18) |
| Fig 7 | 50-episode eval (25 steps) | Fail, 13 of 14 (`paper_start_near_paper`; rep v29) |
<!-- summary:end -->

---

## Fig 3 — GPi α–β oscillation power distribution

Distribution of GPi **α–β** oscillation power (**7–35 Hz**) for **PD On** vs **PD Off** (no DBS). **PD On** = parkinsonian (`pd=1`); **PD Off** = healthy (`pd=0`). Reward threshold θ = 150 is **not** drawn on this panel.

### Paper (Nguyen et al.)

![Paper Fig 3](images/3/paper.png)

### Replication

![Replication Fig 3](images/3/alpha_beta_dist_v22.png)

<!-- caption-3:start -->
**Caption:** GPi α–β (7–35 Hz), 500 iters × 0.1 s; PD On mean=290.8, PD Off mean=219.5, PD On Q1=262.3; ordering_pass=True (v22)

**Manifest:** `artifacts/figures/papers/nguyen/3/manifest.json`
<!-- caption-3:end -->

**Status:** Pass — 500 × 100 ms samples; see `alpha_beta_dist_v22.png`.

<!-- gates-3:start -->
**Gates set** (`artifacts/figures/papers/nguyen/3/manifest.json`; overall **`pass`**: yes, 2026-10-12). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `ordering_pd_on_above_pd_off` | PD On above PD Off | yes |
| `threshold_near_pd_on_q1` | threshold near PD On Q1 (informational) | yes |
| `paper_ordering_pd_on_above_pd_off` | digitization — ordering | yes |
| `paper_mean_ratio_near_paper_readout` | digitization — mean ratio (no curves_fig3 yet) | yes |
| `paper_means_separated` | digitization — means separated (logged) | yes |
<!-- gates-3:end -->

**Run:**

```bash
tmux new-session -d -s fig2-3 \
 "setsid nohup uv run python -m rl_adaptive_dbs.run \
   scripts/figures/papers/nguyen/3/plot.py >> logs/fig2-3.log 2>&1 < /dev/null"
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --n-iterations 50
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --plot-only
```

**Defaults:** 500 iterations per condition, 0.1 s per sample, seeds `0…499`. Writes `alpha_beta_dist_vN.png`.

---

## Fig 4 — Training rewards and lengths

Episode **rewards** (a) and **lengths** (b) over **500** training episodes. Init DBS **40 Hz / 0.3 ms / 300 nA/cm²**; seed **1**; max **25** steps/episode; θ = 150; early stop after $t_u=3$ consecutive steps below θ. Config: `fig4_nguyen_config` — the paper, the authors' published training loop (arXiv 2606.28600), and documented conventions; reasoning and the 13-seed check in [docs/figures/nguyen/4.md](../../docs/figures/nguyen/4.md).

### Paper (Nguyen et al.)

![Paper Fig 4](images/4/paper.png)

### Replication (**v163**, passes every gate)

![Replication Fig 4](images/4/training_reward_length_v163.png)

<!-- caption-4:start -->
**Caption:** DSQN train 500 ep, seed=1; reward ep150+=-27332, len ep150+=9.2; shape_pass=True pass=True (reward shape=True full=True, length shape=True full=True) (v163)

**Manifest:** `artifacts/figures/papers/nguyen/4/manifest.json`
<!-- caption-4:end -->

**Status:** Pass — see manifest gates (`training_reward_length_v163.png`).

From v163 the plant uses distance-based STN recruitment (ρ = 3, `stn_dbs_spread`), a labelled plant deviation adopted on Oct 11 2026: Kumaravelu drives all ten STN neurons with the full DBS current, which keeps the agent's pulse width near 0.3 ms and makes energy fall where the paper's rises. On that plant the paper-silent replay settings were retuned to 64 minibatches per update and a 1,650-transition warm-up. Of 26 seeds, 15 pass every Fig 4 gate and 11 pass every required gate of Figs 4–6; the 13-seed mean curve passes every gate. Shown: seed 1, the lowest-numbered seed passing Figs 4–6 (seed 0 misses Fig 4). See [docs/figures/nguyen/4.md](../../docs/figures/nguyen/4.md) § Retuning for distance-based STN recruitment. Step sizes (5 Hz, 0.1 ms) come from the paper's Fig 6; see [docs/figures/nguyen/6.md](../../docs/figures/nguyen/6.md).

<!-- gates-4:start -->
**Gates set** (`artifacts/figures/papers/nguyen/4/manifest.json`; **`shape_pass`**: yes, **`pass`**: yes, 2026-10-12). **`shape_pass`**: transition timing and ep 0–200 shape. Ship exit: **`pass`** (all required gates, scale-free, anchored to the digitized paper curves). Both subplot groups required; tier-2 rows are logged in the manifest only.

### Reward (panel a) (`shape_pass`: yes | `pass`: yes)

| Key | Description | Shape | Full |
|-----|-------------|-------|------|
| `reward_t50_timing` | 50% progress ep in [paper 71 − 15, 100]; 90% by paper 86 + 25 | yes | yes |
| `reward_pearson_0_200` | smoothed reward vs paper, Pearson r ep 0–200 ≥ 0.90 | yes | yes |
| `reward_pearson_0_500` | smoothed reward vs paper, Pearson r ep 0–500 ≥ 0.93 | — | yes |
| `reward_relative_gain` | (R̄150–500 − R̄0–50) / abs(R̄0–50) ≥ 0.85 (paper 0.974) | — | yes |
| `reward_late_plateau_hold` | min smoothed progress ep 150–500 ≥ 0.80 (paper 0.96) | — | yes |
| `reward_leads_length` | tier 2: reward 10% progress before length (paper ep 26 vs 62) | — | — |
| `reward_scale_raw_d` | tier 2: first-50 mean within ±35% of paper −487.5k (raw d only) | — | — |

### Length (panel b) (`shape_pass`: yes | `pass`: yes)

| Key | Description | Shape | Full |
|-----|-------------|-------|------|
| `length_late_level` | raw mean ep 150–500 / 350–500 within ±25% of paper 8.57 / 8.19 | — | yes |
| `length_t50_timing` | 50% progress ep in [paper 76 − 15, 100]; 90% by paper 89 + 25 | yes | yes |
| `length_pearson_0_200` | smoothed length vs paper, Pearson r ep 0–200 ≥ 0.90 | yes | yes |
| `length_start_at_horizon` | raw median ep 0–50 ≥ horizon − 2 (paper 24.9) | yes | yes |
| `length_pearson_0_500` | smoothed length vs paper, Pearson r ep 0–500 ≥ 0.93 | — | yes |
| `length_ratio` | L̄150–500 / L̄0–50 within ±25% of paper 0.344 | — | yes |
| `length_late_plateau_hold` | min smoothed progress ep 150–500 ≥ 0.80 (paper 0.92) | — | yes |
| `late_timeouts` | timeout share ep 150–500 ≤ 5% (paper 0) | — | yes |
| `late_length_no_regression` | smoothed length slope ep 350–490 ≤ 0.02/ep (paper −0.007) | — | yes |
<!-- gates-4:end -->

`--smoke` sets `smoke_override` (CI only).

**Run:**

```bash
uv run rl-dbs panel run nguyen/4 -- --seed 1 --episodes 500
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --plot-only --export-notes --update-report
```

---

## Fig 5 — Network spikes and DBS energy

Per-episode **CBGT spike counts** (a) and **DBS energy** (b, Eq. (6)) from the same **500**-episode train as Fig 4 (`artifacts/figures/papers/nguyen/4/series.json`).

### Paper (Nguyen et al.)

![Paper Fig 5](images/5/paper.png)

### Replication

![Replication Fig 5](images/5/spikes_energy_v62.png)

<!-- caption-5:start -->
**Caption:** Fig 4 shared train 500 ep, seed=1; spike_mean=53, energy_mean=1269.3; pass=True (v62)

**Manifest:** `artifacts/figures/papers/nguyen/5/manifest.json`
<!-- caption-5:end -->

**Status:** Pass — see manifest gates (`spikes_energy_v62.png`).

From v56 the spike panel is a real count (spike events per 100 ms step across the observed CBGT populations); up to v55 it was a formula anchored on the paper's ~810. Spike and energy levels are report-only; see [docs/figures/nguyen/5.md](../../docs/figures/nguyen/5.md) § Correction and gate split. From v58 the count excludes the stimulated STN, which fires about once per DBS pulse and made the count follow frequency; it comes from a seed-1 re-log that reproduces the v161 train exactly (Fig 4 v162). See [5.md](../../docs/figures/nguyen/5.md) § High-charge STN block. From v59 the paper's traces keep their one-episode dips and each count has its own zero-based axis. Energy is about half the paper's because the paper's agent raised pulse width to ~1.1 ms and ours has no reason to on our plant ([5.md](../../docs/figures/nguyen/5.md) § Paper traces, plot axes, and the energy gap). From v60 the count also leaves out GPi, which follows the STN's pulses; the rest of the network stays flat (late/early ×0.97 vs the paper's ×1.01) while the agent moves from ~24 Hz to ~67 Hz ([5.md](../../docs/figures/nguyen/5.md) § Count without STN and GPi). From v61 the train uses distance-based STN recruitment (Fig 4 v163): the agent widens the pulse to ~0.85 ms late, so energy rises from ~700 to ~1,440 after episode ~70, against the paper's ~2,035 plateau. v62 replots the same train and adds a report-only metric: Eq. (6) at each episode's final setting × 25 steps, which reads 1,739 late (the plotted per-step mean reads 1,437). The paper's late ~2,040 matches its own Fig 6 setting read the same way, so about half the gap is how the episode's energy is formed. The other half is the operating point: ours ends at ~65 Hz / 0.85 ms and the paper's at ~81 Hz / 1.13 ms, and on our plant Eq. (7) can't tell the two apart. See [5.md](../../docs/figures/nguyen/5.md) § Dose-response and the late energy gap.

<!-- gates-5:start -->
**Gates set** (`artifacts/figures/papers/nguyen/5/manifest.json`; overall **`pass`**: yes, 2026-10-12). Rows marked *report only* are logged and never change `pass`; the rest are required.

| Key | Description | Pass |
|-----|-------------|------|
| `shared_train` | Fig 4 passed + same n_episodes | yes |
| `spike_series_has_variance` | spike series has variance | yes |
| `energy_series_has_variance` | energy series has variance | yes |
| `energy_not_constant` | energy not constant | yes |
| `energy_in_paper_band` | mean 300–3200/ep, max ≤ 3520 | yes |
| `paper_spike_trend_near_paper` | digitization — spikes flat trend (late/early) | yes |
| `paper_energy_monotonic_rise` | digitization — monotonic rise early < mid < late | yes |
| `paper_energy_late_above_early` | energy late > early × 1.25 | yes |
| `paper_spike_series_has_variance` | digitization — spike variance | yes |
| `paper_energy_not_constant` | digitization — energy not constant | yes |
| `spike_in_paper_band` | *report only* — mean spikes 400–950/ep | no |
| `paper_spike_early_near_paper` | *report only* — digitization — early spikes near paper (ep 0–50) | no |
| `paper_spike_mid_near_paper` | *report only* — digitization — mid spikes near paper (ep 55–75) | no |
| `paper_spike_late_near_paper` | *report only* — digitization — late spikes near paper (ep 350–500) | no |
| `paper_spike_mean_near_paper` | *report only* — digitization — spike mean near paper | no |
| `paper_spike_stays_near_800` | *report only* — spike count starts & stays ~800 | no |
| `paper_energy_early_near_paper` | *report only* — digitization — early energy near paper (ep 0–50) | no |
| `paper_energy_mid_near_paper` | *report only* — digitization — mid energy near paper (ep 55–75) | no |
| `paper_energy_late_near_paper` | *report only* — digitization — late energy near paper (ep 350–500) | no |
| `paper_energy_mean_near_paper` | *report only* — digitization — energy mean | no |
| `paper_energy_mid_ramp_near_paper` | *report only* — digitization — energy ramp ep 55–75 | yes |
| `paper_energy_trend_near_paper` | *report only* — digitization — energy late vs early | yes |
<!-- gates-5:end -->

**Run:** plot from Fig 4 series cache; `scripts/figures/papers/nguyen/5/plot.py --plot-only` after train.

---

## Fig 6 — α–β and DBS parameters over training

GPi **α–β** (a) and DBS amplitude / frequency / pulse width (b) over **500** episodes; shared train with Figs 4–5.

### Paper (Nguyen et al.)

![Paper Fig 6](images/6/paper.png)

### Replication

![Replication Fig 6](images/6/alpha_beta_params_v18.png)

<!-- caption-6:start -->
**Caption:** Fig 4 shared train 500 ep; αβ_late=91.3, amp=310; pass=True (v18)

**Manifest:** `artifacts/figures/papers/nguyen/6/manifest.json`
<!-- caption-6:end -->

**Status:** Pass — see manifest gates (`alpha_beta_params_v18.png`).

Late pulse width (~0.85 ms vs the paper's 1.13) and frequency (~65 Hz vs 81) sit below the paper's. On our plant α–β stops improving at 0.5–0.6 ms, and Eq. (7) return is flat above ~65 Hz ([docs/figures/nguyen/6.md](../../docs/figures/nguyen/6.md) § Late pulse width and frequency vs the paper).

<!-- gates-6:start -->
**Gates set** (`artifacts/figures/papers/nguyen/6/manifest.json`; overall **`pass`**: yes, 2026-10-12). Rows marked *report only* are logged and never change `pass`; the rest are required.

| Key | Description | Pass |
|-----|-------------|------|
| `shared_train` | Fig 4 passed + shared train | yes |
| `paper_alpha_beta_series_has_variance` | digitization — α–β series has variance | yes |
| `paper_alpha_beta_early_above_theta` | digitization — early α–β > θ=150 (ep 0–50) | yes |
| `paper_alpha_beta_early_near_paper` | digitization — early α–β near paper (ep 0–50) | yes |
| `paper_alpha_beta_mid_near_paper` | digitization — mid α–β near paper (ep 50–100) | yes |
| `paper_alpha_beta_drops_by_100` | digitization — α–β drops by ep 100 | yes |
| `paper_alpha_beta_post100_below_theta` | digitization — post-100 α–β ≤ θ=150 (ep 100–250) | yes |
| `paper_alpha_beta_post100_near_paper` | digitization — post-100 α–β near paper (ep 100–250) | yes |
| `paper_alpha_beta_late_below_theta` | digitization — late α–β ≤ θ=150 (ep 350–500) | yes |
| `paper_alpha_beta_late_near_paper` | digitization — late α–β near paper (ep 350–500) | yes |
| `paper_alpha_beta_mean_near_paper` | digitization — overall α–β mean near paper | yes |
| `paper_alpha_beta_monotonic_drop` | digitization — monotonic drop early > mid > late | yes |
| `paper_alpha_beta_trend_near_paper` | digitization — late vs early α–β ratio | yes |
| `paper_alpha_beta_drop_magnitude_near_paper` | digitization — drop magnitude near paper | yes |
| `paper_alpha_beta_late_stable` | digitization — late α–β stable | yes |
| `paper_alpha_beta_decreases_like_paper` | digitization — α–β decreases | yes |
| `paper_freq_left_init` | frequency >5% off init | yes |
| `paper_freq_late_near_paper` | digitization — late frequency | yes |
| `paper_freq_late_stable` | frequency std last 50 ep ≤ 20% of mean | yes |
| `paper_params_left_init` | *report only* — amp / freq / pw each >5% off init | no |
| `paper_amp_late_near_paper` | *report only* — digitization — late amplitude | yes |
| `paper_pw_late_near_paper` | *report only* — digitization — late pulse width | yes |
| `paper_late_params_stable` | *report only* — std last 50 ep ≤ 20% of mean | no |
| `paper_amp_left_init` | *report only* — amplitude >5% off init | no |
| `paper_pw_left_init` | *report only* — pulse width >5% off init | yes |
<!-- gates-6:end -->

Paper end anchors (~262 nA/cm², ~78.65 Hz, ~1 ms) and ~22% energy vs open-loop 130 Hz are **report** items in manifest metrics, not separate pass keys.

**Run:** shared Fig 4 series; `scripts/figures/papers/nguyen/6/plot.py`.

---

## Fig 7 — Evaluation (50 episodes × 25 steps)

Seeded eval of the trained policy: **50** episodes × **25** steps, different seed per episode.

### Paper (Nguyen et al.)

![Paper Fig 7](images/7/paper.png)

### Replication

![Replication Fig 7](images/7/eval_50ep_v29.png)

<!-- caption-7:start -->
**Caption:** eval 50×26 steps; mean αβ=151.5; pass=False (v29)

**Manifest:** `artifacts/figures/papers/nguyen/7/manifest.json`
<!-- caption-7:end -->

**Status:** Open — see manifest gates (`eval_50ep_v29.png`).

From v24 every point is a plant reading; up to v23 the eval prepended two invented α–β values per episode. See [docs/figures/nguyen/7.md](../../docs/figures/nguyen/7.md) § Correction.

**Adopted (Oct 12 2026): the authors' reset window.** From v29 the eval starts each episode the way the authors' published code does (github.com/howyoubinh/CL-DBS-RL, `MousePDEnv.reset()`): the fresh network's first 100 ms, the step-0 reading, run at 100 Hz / 300 nA/cm² / 0.3 ms, then the agent gets the paper's 40 Hz / 0.3 ms / 300 nA/cm². The paper is silent on what the reset reading runs at, and nothing in the code's reset contradicts it (fresh network and seed per episode, nothing carried over). **Eval only:** `fig4_nguyen_config` and the official checkpoint are unchanged; training with the reset window changed nothing beyond seed noise. On the same checkpoint and seeds, gates go from 11 to 13 of 14: the peak moves to step 2 and $r$ to 0.88. Only `start_near_paper` fails: step 0 reads 197 against the paper's 153 (gate 115–191), because on the ρ = 3 plant the reset pulses recruit less of the STN (the uniform plant reads ~140). `--initial-setting-reset` reproduces the v28 eval. Details: [7.md](../../docs/figures/nguyen/7.md) § Adopted: the authors' reset window.

### Earlier best match (superseded Oct 12 2026, kept for the record)

Superseded by the reset-window eval above, which needs no guesses against stated text. Slip + carry was never adopted: its two episode-start guesses contradict §IV's initial parameters and per-episode seeds ([7.md](../../docs/figures/nguyen/7.md) § Episode-start convention and gate audit).

![Best Fig 7: official checkpoint, slip + carry](images/7/eval_50ep_slipcarry_v2.png)

The official checkpoint (the same one as Figs 4–6 and the run above) with the slip + carry episode start. Passes every Fig 7 gate from plant readings: steps 0–6 = 123, 230, 229, 208, 177, 152, 148 vs the paper's 146, 252, 278, 259, 206, 190, 171; mid 131 vs 161; late 124 vs 153; RMSE 32; $r$ = 0.99. It rests on two guesses about the authors' environment that neither paper states:

- Each episode's first reading runs at the previous episode's final DBS setting.
- Neuron state carries across episodes. The follow-up paper randomizes initial conditions per episode, which argues against this.

The earlier best match (`eval_50ep_slipcarry_v1.png`, seed 4 trained past the $t_u$ stop) also needed a third deviation and failed Fig 4. Details, the reproduce command (`scripts/probes/nguyen_fig7_slip_carry.py`) and the questions for the authors: [7.md](../../docs/figures/nguyen/7.md) § Parked on the slip + carry plot and § Onset on the distance-based plant.

<!-- gates-7:start -->
**Gates set** (`artifacts/figures/papers/nguyen/7/manifest.json`; overall **`pass`**: no, 2026-10-12). Every row is required for exit.

| Key | Description | Pass |
|-----|-------------|------|
| `checkpoint_lineage_ok` | Fig 4 train passed | yes |
| `paper_eval_protocol_ok` | digitization — 50 episodes, each with the reset reading + 25 steps (paper §IV) | yes |
| `paper_step_series_finite` | digitization — step series finite | yes |
| `paper_start_near_paper` | digitization — step 0 within ±25 % of paper (153.2) | no |
| `paper_peak_step_timing` | digitization — peak step within ±1 of paper (2.1) | yes |
| `paper_peak_power_near_paper` | digitization — peak within ±25 % of paper (278.4) | yes |
| `paper_early_mean_near_paper` | digitization — steps 0–5 mean within ±25 % of paper (233.3) | yes |
| `paper_mid_mean_near_paper` | digitization — steps 6–15 mean within ±25 % of paper (161.0) | yes |
| `paper_late_mean_near_paper` | digitization — steps 18–25 mean within ±25 % of paper (152.9) | yes |
| `paper_overall_mean_near_paper` | digitization — overall mean within ±25 % of paper (174.0) | yes |
| `paper_peak_to_late_drop` | digitization — peak − late within ±50 % of paper (125.5) | yes |
| `paper_late_peak_ratio_near_paper` | digitization — late / peak within ±30 % of paper (0.55) | yes |
| `paper_pearson_shape_ok` | digitization — mean-trace Pearson r vs paper ≥ 0.80 | yes |
| `paper_below_fig3_pd_median` | overall mean below Fig 3 PD On (our median and paper ~295) | yes |
<!-- gates-7:end -->

**Run:**

```bash
# Full eval of the Fig 4 checkpoint (50 episodes × 25 steps, 100 Hz reset window), then gates + plot
uv run rl-dbs panel run nguyen/7 -- --export-notes --update-report
# Re-gate and replot from the cached eval
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/7/plot.py --plot-only --export-notes --update-report
```

---

## Conventions

Paper-silent settings of the shipped recipe (`fig4_nguyen_config`; detail and sources in [docs/figures/nguyen/4.md](../../docs/figures/nguyen/4.md) and [snn/replication.md](../../docs/controllers/snn/replication.md)):

| Setting | Choice | Source |
|------|-----------|--------|
| Observation | binary spikes of all eight CBGT populations (80 neurons × 10 bins) | follow-up (arXiv 2606.28600) |
| Action head | `factored` (3× ternary); ties go to the higher output membrane | convention; follow-up uses membrane argmax |
| Steps per action | 5 Hz / 0.1 ms / 10 nA/cm² | frequency and pulse width read off Fig 6; amplitude convention |
| Early-stop streak $t_u$ | 3 | convention |
| Target net, loss, optimizer | soft target τ 0.005, SmoothL1, AdamW, value clip 100 | follow-up |
| Reward δ, τ | 1, 330 (learning rewards × 1e-4) | convention |
| Updates | every 128 transitions (paper), 64 minibatches, 1,650-transition warm-up | paper / convention |
| Plant | one continuous simulation per episode, fresh network per episode; STN recruitment ρ = 3 | follow-up; ρ = 3 approved plant deviation (Oct 11 2026) |
| Eval episode start | fresh network; reset reading at 100 Hz / 300 nA/cm² / 0.3 ms recorded as step 0, then 40 Hz / 0.3 ms / 300 nA/cm² for the agent (eval only, from v29) | paper §IV; authors' published code (`MousePDEnv.reset()`) |
| Fig 3 α–β window | 100 ms per sample (one RL step); 500 iterations | paper |
