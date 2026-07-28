# Nguyen et al. — figure comparisons

**Primary replication tracker** for the DSQN / closed-loop neuromorphic DBS paper. Spec: [controllers/snn/replication.md](../../docs/controllers/snn/replication.md). Work proceeds **in parallel** with Mehregan ([Mehregan replications](../mehregan/replications.md)): each row below is an exit criterion with qualitative gates, a committed `plot.py` (when present), and side-by-side PNGs.

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/nguyen/images/`; JSON caches to `artifacts/figures/papers/nguyen/`.

Figs **1–2** are schematics (CBGT circuit diagram; closed-loop block diagram) — **not** replication targets.

| Panel | Script | Spec | Status |
|-------|--------|------|--------|
| Fig 3 — GPi α–β distribution (PD Off vs PD On) | `scripts/figures/papers/nguyen/3/plot.py` | [snn/replication.md](../../docs/controllers/snn/replication.md) §3 | Pass (layout + ordering + scale) |
| Fig 4 — training reward + episode length | `scripts/figures/papers/nguyen/4/plot.py` | [snn/replication.md](../../docs/controllers/snn/replication.md) §6.5 | Open |
| Fig 5 — CBGT spikes + DBS energy over training | `scripts/figures/papers/nguyen/5/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Open |
| Fig 6 — α–β + DBS params over training | `scripts/figures/papers/nguyen/6/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Open |
| Fig 7 — 50-episode eval (25 steps) | `scripts/figures/papers/nguyen/7/plot.py` (planned) | [snn/replication.md](../../docs/controllers/snn/replication.md) §8 | Open |

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
 "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \
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

**Status:** Open — `DSQNTrainer` + `rl-dbs train --controller snn` exist; needs a real plant train + `scripts/figures/papers/nguyen/4/plot.py`.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 500 episodes; init 40 Hz / 0.3 ms / 300 nA/cm² | TBD | — |
| **Early phase** | High variance ~episodes 0–100 (exploration) | TBD | — |
| **Later phase** | Shift toward exploitation; reward/length stabilize | TBD | — |
| **Seed note** | Paper seed unspecified | Lock one seed for gates | — |

**Run (planned):**

```bash
tmux new-session -d -s fig2-4-train \
 "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \
   scripts/figures/papers/nguyen/4/plot.py >> logs/fig2-4-train.log 2>&1 < /dev/null"
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --plot-only
```

**Defaults (planned):** 500 episodes; seed `0`; early-stop streak $t_u=3$; θ = 150 (once scale is locked or documented).

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

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | Same 500-episode train as Fig 4 | TBD | — |
| **Spikes panel** | CBGT spike counts readable over episodes | TBD | — |
| **Energy panel** | Energy responds as DBS parameters move | TBD | — |
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

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **α–β trend** | Decreases vs PD baseline / toward or below θ | TBD | — |
| **Params leave init** | Leave 40 Hz / 0.3 ms / 300 nA/cm² | TBD | — |
| **Therapeutic band** | Settle near clinical-ish settings | TBD | — |
| **Paper readouts** | ~262 nA/cm², ~78.65 Hz, ~1 ms | Optional after shape | — |

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

**Status:** Open — needs a locked Fig 4–6 checkpoint + `rl-dbs eval --controller snn` + panel script.

### Side-by-side checklist

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Protocol** | 50 episodes × 25 steps; varying seeds | TBD | — |
| **α–β vs untreated** | Mean under policy **below** Fig 3 PD reference | TBD | — |
| **Stability** | Readable mean ± spread over steps | TBD | — |

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
