# Ravivarapu et al. — figure comparisons

**Primary replication tracker** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Work is scheduled by **panel**, not by roadmap phase: each row below is an exit criterion with qualitative gates, a committed `plot.py` (planned until Phase 6 lands), and side-by-side PNGs.

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/ravivarapu/images/`; JSON caches to `artifacts/figures/papers/`.

Controller / adapter spec: [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md). Shared plant: [plant.md](../../docs/plant.md). Schematics (paper Figs 1, 3) are **out of scope** for this tracker. Fig 2 (reward curve of Eq. (7)) is optional polish, not a gate.

| Panel | Script | Spec | Status |
|-------|--------|------|--------|
| Fig 4a — training PSD vs episode | `scripts/figures/papers/ravivarapu/4a/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |
| Fig 4b — training reward vs episode | `scripts/figures/papers/ravivarapu/4b/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |
| Fig 5a — inference @ 50 Hz | `scripts/figures/papers/ravivarapu/5a/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |
| Fig 5b — inference @ 30 Hz | `scripts/figures/papers/ravivarapu/5b/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |
| Fig 6 — FP16 PTQ @ 50 Hz | `scripts/figures/papers/ravivarapu/6/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |
| Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS) | `scripts/figures/papers/ravivarapu/7/plot.py` (planned) | [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) | Open |

Replication PNGs: `figures/ravivarapu/images/`. JSON caches: `artifacts/figures/papers/`. Paper crops: `figures/ravivarapu/images/<panel>/paper.png` (from KB paper-note embeds; Fig 4/5 split from combined panels). Full composites also under `figures/ravivarapu/images/_full/`.

**Blocking:** `controllers/sea_dbs/` is placeholder-only; panels need the adapter (2 ms steps, binary pulse, Eq. (7) reward) and trainer variants before train/plot scripts can run. See [sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) §5–§12.

---

## Fig 4a — training PSD vs episode

Average **beta-band PSD** across **training episodes** for **Baseline (DDPG)** vs **SEA-DBS** (§V.B / Fig. 4(a)). Paper claim: SEA-DBS shows a **more pronounced and consistent** beta suppression over episodes; Baseline declines only modestly.

Related numeric protocol (not a separate panel): **Table II** — seed change every $n \in \{10, 20, 50, 75\}$ steps; SEA-DBS lower avg PSD and higher avg reward than Baseline at each $n$. Track with `sea_dbs_eval` / harness once training exists.

### Paper (Ravivarapu et al.)

![Paper Fig 4a](images/4a/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/4a/training_psd.png`

<!-- caption-4a:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/4a/manifest.json` (planned)
<!-- caption-4a:end -->

**Status:** Open — needs SEA-DBS trainer + Baseline ablation.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Series** | Baseline vs SEA-DBS | TBD | — |
| **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** / steeper drop over episodes | TBD | — |
| **Episode axis** | Training episodes (paper panel) | TBD (lock episode count; paper uses **150** train episodes) | — |
| **Qualitative shape** | SEA-DBS clearer suppression; Baseline modest | TBD | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --plot-only
```

**Defaults (paper §V.A / Table I):** seed `0` (paper seed unspecified — lock one seed for gates), **2 ms** RL step × **30** steps/episode × **150** episodes, $\gamma=0.99$, buffer **8192**, batch **32**, $\alpha_a=5\times10^{-4}$, $\alpha_c=10^{-3}$, variants `baseline` vs `paper`.

---

## Fig 4b — training reward vs episode

**Cumulative / episode reward** during the same training comparison as Fig 4a (§V.B / Fig. 4(b)). Paper claim: SEA-DBS reaches **higher** rewards with a **faster** early rise than Baseline.

### Paper (Ravivarapu et al.)

![Paper Fig 4b](images/4b/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/4b/training_reward.png`

<!-- caption-4b:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/4b/manifest.json` (planned)
<!-- caption-4b:end -->

**Status:** Open — pair with Fig 4a locked run.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Series** | Baseline vs SEA-DBS | TBD | — |
| **SEA-DBS vs Baseline reward** | SEA-DBS **higher**, faster early rise | TBD | — |
| **Paired with Fig 4a** | Same training run | TBD | — |
| **Trend** | Reward↑ as PSD↓ (with Fig 4a) | TBD | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py --plot-only
```

**Defaults:** paired Fig 4a cache / checkpoint; seed `0`; reward from Eq. (7) ($\beta_t = 0.35$).

---

## Fig 5a — inference @ 50 Hz

Post-train **inference** comparison of SEA-DBS vs Baseline with stimulation **carrier frequency 50 Hz** (above beta band; Fig. 5(a)). Paper claim: **50 Hz** more effectively disrupts pathological oscillations → **greater PSD reduction** than the 30 Hz panel; SEA-DBS **below** Baseline on PSD (and higher reward).

Carrier frequency is a **fixed eval setting**, not a per-step RL action ([sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) §14.10).

### Paper (Ravivarapu et al.)

![Paper Fig 5a](images/5a/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/5a/inference_50hz.png`

<!-- caption-5a:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/5a/manifest.json` (planned)
<!-- caption-5a:end -->

**Status:** Open — needs trained `paper` + `baseline` actors and adapter carrier-frequency knob.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Carrier** | 50 Hz fixed at inference | TBD | — |
| **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** | TBD | — |
| **vs Fig 5b** | 50 Hz **stronger** suppression than 30 Hz | TBD | — |
| **Protocol length** | Inference traces as in paper panel | TBD (document step count / window) | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5a/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5a/plot.py --plot-only
```

**Defaults:** seed `0`; carrier **50 Hz**; binary pulse policy from trained SEA-DBS / Baseline.

---

## Fig 5b — inference @ 30 Hz

Same inference layout at **30 Hz** carrier (overlaps pathological beta; Fig. 5(b)). Paper claim: **less effective** than 50 Hz; SEA-DBS still **beats** Baseline on PSD / reward.

### Paper (Ravivarapu et al.)

![Paper Fig 5b](images/5b/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/5b/inference_30hz.png`

<!-- caption-5b:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/5b/manifest.json` (planned)
<!-- caption-5b:end -->

**Status:** Open — pair protocol with Fig 5a; only carrier differs.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Carrier** | 30 Hz fixed at inference | TBD | — |
| **SEA-DBS vs Baseline PSD** | SEA-DBS **lower** | TBD | — |
| **vs Fig 5a** | 30 Hz **weaker** suppression than 50 Hz | TBD | — |
| **Shared protocol** | Same eval length / seeding as Fig 5a | TBD | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5b/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5b/plot.py --plot-only
```

**Defaults:** seed `0`; carrier **30 Hz**; same checkpoints as Fig 5a where possible.

---

## Fig 6 — FP16 PTQ @ 50 Hz

**FP16 post-training quantization** of SEA-DBS vs Baseline at **50 Hz** over **10 stimulation steps** (Fig. 6 / §V). Paper claim: quantized SEA-DBS **tracks** full-precision PSD reduction and still **beats** Baseline; model size **~65 MB → ~33 MB**.

QAT is **out of scope** for SEA-DBS (not reported).

### Paper (Ravivarapu et al.)

![Paper Fig 6](images/6/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/6/ptq_fp16_50hz.png`

<!-- caption-6:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/6/manifest.json` (planned)
<!-- caption-6:end -->

**Status:** Open — needs FP16 PTQ path on SEA-DBS actor after full-precision train.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Steps** | **10** stimulation steps | TBD | — |
| **PTQ vs FP32 (SEA-DBS)** | Track closely | TBD | — |
| **SEA-DBS PTQ vs Baseline** | SEA-DBS still **lower** PSD | TBD | — |
| **Carrier** | 50 Hz | TBD | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/6/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/6/plot.py --plot-only
```

**Defaults:** seed `0`; **FP16 PTQ** only (no QAT); 10-step eval; 50 Hz carrier.

---

## Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS)

**PSD over 10 stimulation steps** for four variants (Fig. 7): **Baseline**, **Baseline+PM**, **Baseline+GS**, **SEA-DBS (PM+GS)**. Paper claim: SEA-DBS strongest / most stable suppression; +PM alone noisy early (~**4,500** samples cited); +GS alone limited gains.

Map to trainer `variant`: `baseline`, `baseline-pm`, `baseline-gs`, `paper` ([sea_dbs/replication.md](../../docs/controllers/sea_dbs/replication.md) §12).

### Paper (Ravivarapu et al.)

![Paper Fig 7](images/7/paper.png)

### Replication

*Not yet generated.* Target: `figures/ravivarapu/images/7/ablation_psd.png`

<!-- caption-7:start -->
**Caption:** TBD

**Manifest:** `artifacts/figures/papers/ravivarapu/7/manifest.json` (planned)
<!-- caption-7:end -->

**Status:** Open — needs all four variants trainable and a shared 10-step eval harness.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Four series** | Baseline, +PM, +GS, SEA-DBS | TBD | — |
| **Ordering** | SEA-DBS **lowest** / most stable PSD | TBD | — |
| **+PM early** | Noisy / limited early (small sample regime) | TBD | — |
| **Steps** | **10** stimulation steps | TBD | — |

**Run (planned):**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/7/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/7/plot.py --plot-only
```

**Defaults:** seed `0`; variants `baseline`, `baseline-pm`, `baseline-gs`, `paper`; 10-step PSD eval.
