# Nguyen et al. — figure comparisons

**Primary replication tracker** for the DSQN / closed-loop neuromorphic DBS paper. Spec: [controllers/snn/replication.md](../../docs/controllers/snn/replication.md). Work proceeds **in parallel** with Mehregan ([Mehregan replications](../mehregan/replications.md)): each row below is an exit criterion with qualitative gates, a committed `plot.py` (when present), and side-by-side PNGs.

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/nguyen/images/`; JSON caches to `artifacts/figures/papers/nguyen/`.

Figs **1–2** are schematics (CBGT circuit diagram; closed-loop block diagram) — **not** replication targets.

| Panel | Script | Spec | Gates | Status |
|-------|--------|------|-------|--------|
| Fig 3 — GPi α–β distribution (PD Off vs PD On) | `scripts/figures/papers/nguyen/3/plot.py` | [snn/replication.md](../../docs/controllers/snn/replication.md) §3 | Set (§ below) | Pass |
| Fig 4 — training reward + episode length | `scripts/figures/papers/nguyen/4/plot.py` | [snn/replication.md](../../docs/controllers/snn/replication.md) §6.5 | Set (§ below) | Open |
| Fig 5 — CBGT spikes + DBS energy over training | `scripts/figures/papers/nguyen/5/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Set (§ below) | Open |
| Fig 6 — α–β + DBS params over training | `scripts/figures/papers/nguyen/6/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Set (§ below) | Open |
| Fig 7 — 50-episode eval (25 steps) | `scripts/figures/papers/nguyen/7/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Set (§ below) | Open |

Replication PNGs: `figures/nguyen/images/`. JSON caches: `artifacts/figures/papers/nguyen/`. Paper panels: `figures/nguyen/images/<panel>/paper.png` (exact copies of paper-note embeds). Full copies also under `figures/nguyen/images/_full/`.

---

## Fig 3 — GPi α–β oscillation power distribution

Distribution of GPi **α–β** oscillation power (**7–35 Hz**: α 7–13 Hz + β 13–35 Hz) for **PD On** vs **PD Off** (no DBS). Paper panel: **(a)** per-iteration scatter with mean reference lines, **(b)** boxplot summary. Control threshold **θ = 150** (§IV reward) is chosen near the PD On first quartile — it is **not** drawn on Fig 3.

Unlike Mehregan $P_\beta$ (13–35 Hz only), Nguyen’s feedback band is the full **7–35 Hz** α–β index — see [snn/replication.md](../../docs/controllers/snn/replication.md) §3. Exact PSD estimator is **intentionally open** in the paper; v1 uses the repo `alpha_beta_power` helper on GPi spikes with a **100 ms** integration window (Nguyen RL step, §IV).

### Paper (Nguyen et al.)

![Paper Fig 3](images/3/paper.png)

### Replication

![Replication Fig 3](images/3/alpha_beta_dist_v4.png)

<!-- caption-3:start -->
**Caption:** GPi α–β (7–35 Hz), 500 iters × 0.1 s; PD On mean=290.8, PD Off mean=219.5, PD On Q1=262.3; ordering_pass=True (v4)

**Manifest:** `artifacts/figures/papers/nguyen/3/manifest.json`
<!-- caption-3:end -->

**Status:** Pass — 500 × 100 ms samples; layout matches paper (iteration scatter, mean PD Off/On lines, boxplot). Scatter marker size 22. Panel **(a)** legend uses a translucent framed key (`framealpha=0.75`). Means ~220 / ~291 vs paper ~215 / ~295.

**Qualitative gates (paper Fig 3 — exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Layout** | **(a)** per-iteration scatter + mean reference lines; **(b)** boxplot; labels **PD Off** / **PD On** | Missing panel, wrong labels, or no boxplot |
| 2 | **Ordering** | **PD On** (parkinsonian) mass **above** **PD Off** (healthy / no-PD) | `median(PD On) ≤ median(PD Off)` |
| 3 | **No DBS** | Untreated distributions only | Any STN stimulation during sampling |
| 4 | **Scale (qualitative)** | Means roughly ~215 / ~295 (paper read) | Ordering passes but both means sit in the same band (no separation) |
| 5 | **θ plausibility (soft)** | Paper picks θ = 150 near PD On Q1; **not drawn** on this panel | `|PD On Q1 − 150| / 150 > 0.75` — informational only; does **not** block pass |

**Terminology:** **PD On** = parkinsonian (`pd=1`); **PD Off** = healthy / no-PD (`pd=0`). Matches the paper’s “PD and no-PD” panel (higher α–β in PD).

**Automated mirrors** (`scripts/figures/papers/nguyen/3/plot.py` → manifest `gates`): `ordering_pd_on_above_pd_off`; soft `threshold_near_pd_on_q1`. **`pass`** = gate **2** only (gates 4–5 are qualitative / soft).

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Layout** | (a) scatter + mean lines; (b) boxplot | Same | Yes |
| **Sample count** | ~500 simulation iterations | `--n-iterations 500` (default) | Yes |
| **Labels** | PD Off / PD On | PD Off / PD On | Yes |
| **Mean lines** | Red (PD Off), black (PD On) | Same | Yes |
| **Legend (a)** | Translucent framed key, lower right | `frameon=True`, `framealpha=0.75`, gray border | Yes |
| **Ordering** | PD On **above** PD Off | PD On mean 290.8 > PD Off 219.5 | Yes |
| **Scale** | Means ~215 / ~295 | 219.5 / 290.8 | Yes (qualitative) |
| **θ on panel** | Not drawn | Not drawn (θ=150 is reward-only) | Yes |
| **No DBS** | Untreated distributions | `DbsSpec.none()` | Yes |

Panel **(b)** open circles are **outliers** — individual iterations whose α–β power falls outside the box whiskers (typically >1.5× the interquartile range from the box edge). The paper panel shows the same pattern for low PD On samples.

**Run:**

```bash
tmux new-session -d -s fig2-3 \
 "setsid nohup uv run python -m rl_adaptive_dbs.run \
   scripts/figures/papers/nguyen/3/plot.py >> logs/fig2-3.log 2>&1 < /dev/null"
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --n-iterations 50
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --plot-only
```

**Defaults:** **500** iterations per condition, **0.1 s** (100 ms) integration per sample, Python plant, seeds `0…499`. Each run writes `alpha_beta_dist_vN.png` (N auto-increments).

---

## Fig 4 — Training rewards and lengths

Episode **rewards** (a) and **lengths** (b) over **500** training episodes (§IV; Fig. 4). Each episode starts from init DBS **40 Hz / 0.3 ms / 300 nA/cm²**. Paper claim: high variance (exploration) roughly episodes **0–100**, then a shift toward exploitation / optimization.

### Paper (Nguyen et al.)

![Paper Fig 4](images/4/paper.png)

### Replication

*Not yet generated.* Target: `figures/nguyen/images/4/training_reward_length.png`

<!-- caption-4:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/nguyen/4/manifest.json` (planned)
<!-- caption-4:end -->

**Status:** Open — `DSQNTrainer` + `scripts/figures/papers/nguyen/4/plot.py` exist; needs a full **500**-episode plant train that passes gates below.

**Qualitative gates (paper Fig 4 — exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Protocol** | **500** episodes; init **40 Hz / 0.3 ms / 300 nA/cm²**; seed **0** (paper seed unspecified) | Wrong episode count, init triple, or unlocked seed across promote runs |
| 2 | **Early exploration** | Noisy rewards ~episodes **0–100**; episode lengths near horizon while exploring | Median length in first **50** episodes `< max_steps − 2` (default max **25**) |
| 3 | **Reward improves** | Later episodes beat early exploration returns | Mean reward over episodes **150–500** ≤ mean over first **50** episodes |
| 4 | **Length drops** | Episode length shortens as α–β sub-threshold early termination kicks in | Mean length over episodes **150–500** ≥ mean over first **75** episodes − **1** step |
| 5 | **Exploitation shape (qualitative)** | Smoother / upward reward trend after ~episode **100** | Reward still pure noise with no late uplift (human check on smoothed trace) |

**Automated mirrors** (`evaluate_gates` in `scripts/figures/papers/nguyen/4/plot.py` → manifest `gates`): `early_near_max_length`, `late_reward_above_early`, `length_decreases`; also logs `early_high_variance` (informational). **`pass`** = gates **2–4** (all three booleans). `--smoke` sets `smoke_override` for CI only.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 500 episodes; init 40 Hz / 0.3 ms / 300 nA/cm² | seed `0`; `fig4_nguyen_config` | — |
| **Early phase** | High variance ~episodes 0–100 (exploration) | `early_high_variance` + near-max lengths | — |
| **Later phase** | Shift toward exploitation; reward/length stabilize | `late_reward_above_early` + `length_decreases` | — |
| **Seed note** | Paper seed unspecified | Lock seed `0` for gates | — |

**Run:**

```bash
tmux new-session -d -s fig2-4-train \
 "setsid nohup uv run python -m rl_adaptive_dbs.run \
   scripts/figures/papers/nguyen/4/plot.py >> logs/fig2-4-train.log 2>&1 < /dev/null"
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --plot-only
```

**Defaults:** 500 episodes; seed `0`; early-stop streak $t_u=3$; θ = 150; max **25** steps/episode unless early-terminated.

---

## Fig 5 — Network spikes and DBS energy

Per-episode **CBGT spike counts** (a) and approximate **DBS energy** (b, Eq. (6)) over the same **500** training episodes (§IV; Fig. 5).

### Paper (Nguyen et al.)

![Paper Fig 5](images/5/paper.png)

### Replication

*Not yet generated.* Target: `figures/nguyen/images/5/spikes_energy.png`

<!-- caption-5:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/nguyen/5/manifest.json` (planned)
<!-- caption-5:end -->

**Status:** Open — paired with Fig 4 training logs once the panel script exists.

**Qualitative gates (paper Fig 5 — exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Shared train** | Same **500**-episode DSQN run as Fig 4 (same checkpoint / `series.json`) | Different seed, episode count, or init triple than Fig 4 |
| 2 | **Spikes panel** | CBGT **total spike counts** per episode traceable over training | Flat / missing spike series |
| 3 | **Energy panel** | Eq. (6) **DBS energy** per episode responds as parameters move | Energy constant (±1%) across all episodes |
| 4 | **Co-variation** | Both panels show episode-level structure (not a single scalar repeated) | Either series has zero variance |
| 5 | **Shape vs digits** | Rough downward / settling trends per paper panel | N/A for automated pass — qualitative only |

**Planned automated mirrors:** reuse Fig 4 train manifest; `spike_series_has_variance`, `energy_series_has_variance`, `energy_not_constant`. **`pass`** = gates **1–4**.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | Same 500-episode train as Fig 4 | Shared `artifacts/figures/papers/nguyen/4/series.json` | — |
| **Spikes panel** | CBGT spike counts readable over episodes | Gate 2 | — |
| **Energy panel** | Energy responds as DBS parameters move | Gate 3 | — |
| **Shape vs digits** | Rough trend / ordering | Not digit match | — |

**Run (planned):** same train cache as Fig 4; `scripts/figures/papers/nguyen/5/plot.py` (or shared `--plot-only` from Fig 4 artifacts).

**Defaults (planned):** Eq. (6) energy on STN DBS current; Python plant.

---

## Fig 6 — α–β and DBS parameters over training

GPi **α–β** (a) and the three **DBS parameters** — amplitude, frequency, pulse width — (b) over **500** episodes (§IV; Fig. 6). Paper end-of-training anchors (~**262 nA/cm²**, ~**78.65 Hz**, ~**1 ms**) are **evaluation anchors**, not hard gates until one-seed qualitative shape passes. Paper also claims ~**22%** energy reduction vs open-loop **130 Hz** (report after shape passes).

### Paper (Nguyen et al.)

![Paper Fig 6](images/6/paper.png)

### Replication

*Not yet generated.* Target: `figures/nguyen/images/6/alpha_beta_params.png`

<!-- caption-6:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/nguyen/6/manifest.json` (planned)
<!-- caption-6:end -->

**Status:** Open.

**Qualitative gates (paper Fig 6 — exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Shared train** | Same **500**-episode run as Figs 4–5 | Mismatched checkpoint / series vs Fig 4 |
| 2 | **α–β suppression** | GPi α–β **decreases** over training vs early episodes | Mean α–β over last **100** episodes ≥ mean over first **50** |
| 3 | **Sub-threshold** | Late training α–β at or **below** θ = **150** (raw scale) | Mean α–β over last **100** episodes **> θ** |
| 4 | **Params leave init** | Amplitude, frequency, pulse width move off **300 nA/cm² / 40 Hz / 0.3 ms** | All three params within **5%** of init at episode **500** |
| 5 | **Late stabilization** | DBS parameters plateau in late training (not perpetual limit-cycle) | Each parameter’s std over last **50** episodes **> 20%** of its mean (noise-dominated) |
| 6 | **Paper anchors (soft)** | End state ~**262 nA/cm²**, ~**78.65 Hz**, ~**1 ms** | Informational after gates **2–5** pass — not hard fail |
| 7 | **Energy claim (report)** | ~**22%** lower DBS energy vs open-loop **130 Hz** (Eq. (6)) | Report in caption/manifest after shape passes — not a panel pass gate |

**Planned automated mirrors:** `alpha_beta_decreases`, `late_alpha_beta_below_theta`, `params_left_init`, `late_params_stable`. **`pass`** = gates **2–5**. Gate **6** logged as `paper_anchor_delta_*`.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **α–β trend** | Decreases vs PD baseline / toward or below θ | Gates 2–3 | — |
| **Params leave init** | Leave 40 Hz / 0.3 ms / 300 nA/cm² | Gate 4 | — |
| **Therapeutic band** | Settle near clinical-ish settings | Gate 5 + soft gate 6 | — |
| **Paper readouts** | ~262 nA/cm², ~78.65 Hz, ~1 ms | Soft gate 6 | — |

**Run (planned):** train cache shared with Figs 4–5; `scripts/figures/papers/nguyen/6/plot.py`.

**Defaults (planned):** init triple as Fig 4; factored ternary actions (v1).

---

## Fig 7 — Evaluation (50 episodes × 25 steps)

Seeded eval of the trained policy: average **50** test episodes across **25** time steps with **different seeds** per episode (§IV; Fig. 7). Paper claim: learned policy keeps α–β below untreated PD reference from Fig 3.

### Paper (Nguyen et al.)

![Paper Fig 7](images/7/paper.png)

### Replication

*Not yet generated.* Target: `figures/nguyen/images/7/eval_50ep.png`

<!-- caption-7:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/nguyen/7/manifest.json` (planned)
<!-- caption-7:end -->

**Status:** Open — needs a Fig 4–6 checkpoint that passes training gates + `rl-dbs eval --controller snn` + panel script.

**Qualitative gates (paper Fig 7 — exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Eval protocol** | **50** test episodes × **25** steps; **different seed per episode** | Wrong episode/step counts or shared seed across episodes |
| 2 | **vs Fig 3 PD** | Policy α–β **below** untreated **PD On** reference from Fig 3 (parkinsonian, no DBS) | Mean policy α–β over eval **≥** Fig 3 `PD On` median (from locked Fig 3 manifest) |
| 3 | **vs θ** | Sustained sub-threshold control | Mean policy α–β over all steps **> θ = 150** |
| 4 | **Stability** | Readable mean trace with bounded spread across **25** steps | NaNs, or step-to-step mean range **> 2×** Fig 3 PD On IQR |
| 5 | **Checkpoint lineage** | Eval uses the same trained policy as Figs 4–6 | Checkpoint path ≠ Fig 4 `checkpoint.pt` (or documented promoted ckpt) |

**Planned automated mirrors:** `eval_protocol_ok`, `below_fig3_pd_median`, `below_theta`, `step_series_finite`, `checkpoint_matches_fig4`. **`pass`** = gates **1–4**.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 50 episodes × 25 steps; varying seeds | Gate 1 | — |
| **α–β vs untreated** | Mean under policy **below** Fig 3 PD reference | Gates 2–3 | — |
| **Stability** | Readable mean ± spread over steps | Gate 4 | — |

**Run (planned):**

```bash
uv run rl-dbs eval --controller snn --checkpoint artifacts/snn/<ckpt>.pt
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/7/plot.py --plot-only
```

**Defaults (planned):** 50 episodes, 25 steps, Python plant.

---

## Conventions (v1)

Documented choices for paper-silent knobs (see [snn/replication.md](../../docs/controllers/snn/replication.md) §12 and `SNNConfig`):

| Knob | v1 choice |
|------|-----------|
| Observation | GPi-only binary spike matrix |
| Action head | `factored` (3× ternary argmax) |
| Early-stop streak $t_u$ | 3 |
| LIF θ_th | 1.0 |
| Target net | hard copy on a fixed period |
| Reward coeffs δ, τ | `energy_penalty=0.01`, `threshold_reward=1.0` |
| Fig 3 α–β window | **100 ms** per sample (Nguyen RL step); 500 iterations |
