# Ravivarapu et al. — figure replication gates

**Primary gate spec** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Ship status, side-by-side PNGs, and run commands live in [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md).

Controller / adapter: [sea_dbs/replication.md](../../controllers/sea_dbs/replication.md). Shared plant: [plant.md](../../plant.md). Schematics (paper Figs 1–3) are **out of scope**. Fig 2 (reward curve of Eq. (7)) is optional polish, not a gate.

**Defaults (paper §V.A / Table I):** seed `0` (paper seed unspecified — lock one seed for gates), **2 ms** RL step × **30** steps/episode × **150** training episodes, $\gamma=0.99$, buffer **8192**, batch **32**, $\alpha_a=5\times10^{-4}$, $\alpha_c=10^{-3}$. Variants: `baseline` (DDPG, no PM/GS) vs `paper` (full SEA-DBS).

| Panel | Script (planned) | Status |
|-------|------------------|--------|
| Fig 4a — training PSD vs episode | `scripts/figures/papers/ravivarapu/4a/plot.py` | Open |
| Fig 4b — training reward vs episode | `scripts/figures/papers/ravivarapu/4b/plot.py` | Open |
| Fig 5a — inference @ 50 Hz | `scripts/figures/papers/ravivarapu/5a/plot.py` | Open |
| Fig 5b — inference @ 30 Hz | `scripts/figures/papers/ravivarapu/5b/plot.py` | Open |
| Fig 6 — FP16 PTQ @ 50 Hz | `scripts/figures/papers/ravivarapu/6/plot.py` | Open |
| Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS) | `scripts/figures/papers/ravivarapu/7/plot.py` | Open |

**Blocking:** `controllers/sea_dbs/` is placeholder-only until the adapter (2 ms steps, binary pulse, Eq. (7) reward) and trainer variants land.

---

## Fig 4a — training PSD vs episode

Average **beta-band PSD** across **training episodes** for **Baseline (DDPG)** vs **SEA-DBS** (§V.B / Fig. 4(a)).

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Series** | Baseline vs SEA-DBS (two curves) | Missing either series or wrong variant labels |
| 2 | **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** episode-mean beta PSD over training | SEA-DBS ≥ Baseline at most episodes or flat ordering |
| 3 | **Suppression trend** | SEA-DBS shows **steeper, more consistent** decline; Baseline only **modest** decline | SEA-DBS flat/noisy without downward trend, or Baseline matches SEA-DBS slope |
| 4 | **Episode axis** | Training episodes (paper uses **150**) | Wrong episode count or unlabeled x-axis |
| 5 | **Paired with Fig 4b** | Same training run / checkpoint lineage | 4a and 4b from mismatched trains |

**Automated mirrors (future `plot.py`):** `sea_dbs_psd_below_baseline`; `sea_dbs_psd_trend_down`; episode count = 150; variant ∈ `{baseline, paper}`.

**Related (not a separate panel gate):** Table II seed-change protocol ($n \in \{10,20,50,75\}$ steps) — track via `sea_dbs_eval` once training exists.

---

## Fig 4b — training reward vs episode

**Cumulative / episode reward** during the same training comparison as Fig 4a (§V.B / Fig. 4(b)).

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Series** | Baseline vs SEA-DBS | Missing either series |
| 2 | **SEA-DBS vs Baseline reward** | SEA-DBS **higher** cumulative/episode reward | SEA-DBS ≤ Baseline over most of training |
| 3 | **Early learning** | SEA-DBS **faster initial rise** than Baseline | Baseline catches up immediately or leads early |
| 4 | **Paired with Fig 4a** | Same training run as Fig 4a | Different seeds, hyperparameters, or checkpoints |
| 5 | **Reward–PSD consistency** | Reward ↑ as PSD ↓ (with Fig 4a) | Opposite trends with no documented convention break |

**Automated mirrors:** `sea_dbs_reward_above_baseline`; `sea_dbs_early_rise_faster`; reward from Eq. (7) with $\beta_t=0.35$.

---

## Fig 5a — inference @ 50 Hz

Post-train **inference** comparison at **50 Hz** stimulation carrier (Fig. 5(a)). Carrier frequency is a **fixed eval setting**, not a per-step RL action ([sea_dbs/replication.md](../../controllers/sea_dbs/replication.md) §14.10).

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Carrier** | **50 Hz** fixed at inference | Wrong carrier or conflated with RL action |
| 2 | **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** beta PSD than Baseline | SEA-DBS ≥ Baseline on episode/step means |
| 3 | **vs Fig 5b (50 > 30)** | **50 Hz** yields **stronger** suppression than 30 Hz panel | 30 Hz panel beats 50 Hz when protocols match except carrier |
| 4 | **Trained actors** | Both series use post-train policies from Fig 4 train | Untrained or mismatched checkpoints |
| 5 | **Protocol** | Inference trace length/window documented and locked | Unstated eval length or drifting window |

**Automated mirrors:** `carrier_hz == 50`; `sea_dbs_inference_psd_below_baseline`; optional cross-check `psd_50hz < psd_30hz` once Fig 5b exists.

---

## Fig 5b — inference @ 30 Hz

Same inference layout at **30 Hz** carrier (Fig. 5(b)).

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Carrier** | **30 Hz** fixed at inference | Wrong carrier |
| 2 | **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** than Baseline | SEA-DBS ≥ Baseline |
| 3 | **vs Fig 5a (30 < 50)** | **Weaker** suppression than 50 Hz panel | 30 Hz beats 50 Hz with shared protocol |
| 4 | **Shared protocol** | Same eval length, seeding, checkpoints as Fig 5a | Only carrier differs from 5a |
| 5 | **Biological read** | 30 Hz overlaps pathological beta band → less effective | Claim 30 Hz “better” without convention note |

**Automated mirrors:** `carrier_hz == 30`; `sea_dbs_inference_psd_below_baseline`; `psd_30hz > psd_50hz` (weaker suppression).

---

## Fig 6 — FP16 PTQ @ 50 Hz

**FP16 post-training quantization** of SEA-DBS vs Baseline at **50 Hz** over **10 stimulation steps** (Fig. 6 / §V). QAT is **out of scope**.

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Eval length** | **10** stimulation steps | Wrong step count |
| 2 | **Carrier** | **50 Hz** | Wrong carrier |
| 3 | **PTQ tracks FP32 (SEA-DBS)** | Quantized SEA-DBS **closely tracks** full-precision SEA-DBS PSD path | PTQ diverges to Baseline band or opposite trend |
| 4 | **SEA-DBS PTQ vs Baseline** | SEA-DBS PTQ still **beats** Baseline on PSD | PTQ SEA-DBS ≥ Baseline |
| 5 | **Series labels** | Baseline, SEA-DBS FP32, SEA-DBS FP16 PTQ (as paper panel) | Missing PTQ line or wrong quantization mode |

**Report (not a visual gate):** model size **~65 MB → ~33 MB** after FP16 PTQ.

**Automated mirrors:** `n_steps == 10`; `ptq_tracks_fp32_band`; `sea_dbs_ptq_below_baseline`.

---

## Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS)

**PSD over 10 stimulation steps** for four variants (Fig. 7). Map to trainer `variant`: `baseline`, `baseline-pm`, `baseline-gs`, `paper`.

**Qualitative gates (exit criteria):**

| # | Gate | Paper look | Fail if |
|---|------|------------|---------|
| 1 | **Four series** | Baseline, Baseline+PM, Baseline+GS, SEA-DBS (PM+GS) | Missing variant or wrong ablation mapping |
| 2 | **Ordering** | SEA-DBS **lowest / most stable** PSD | Another variant clearly below SEA-DBS |
| 3 | **+PM early noise** | Baseline+PM **noisy / limited** early (~**4,500** samples cited) | +PM identical to full SEA-DBS from step 0 |
| 4 | **+GS alone** | Baseline+GS **limited** gains over Baseline | +GS matches full SEA-DBS without PM |
| 5 | **Eval length** | **10** stimulation steps | Wrong step count |
| 6 | **Shared train budget** | Comparable training episodes / Table I hyperparameters across variants | Unfair train length or hparams |

**Automated mirrors:** four variants present; `ordering_sea_dbs_lowest`; `n_steps == 10`.

---

## Conventions (v1)

| Knob | v1 choice |
|------|-----------|
| Seed | `0` (lock for one-seed gates) |
| RL step | **2 ms** × **30** steps/episode |
| Training episodes | **150** |
| Reward | Eq. (7), $\beta_t = 0.35$ |
| State input | Mean $\bar{P}_\beta$ over observation window (default until adapter documents $n_{\mathrm{obs}}$) |
| Baseline variant | `baseline` — DDPG without PM or GS |
| Full SEA-DBS | `paper` — PM + GS |
| Inference carrier | Fixed **50 Hz** or **30 Hz** eval knob (not RL action) |
| PTQ | **FP16 PTQ only** (no QAT) |
