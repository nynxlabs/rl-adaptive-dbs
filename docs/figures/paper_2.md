# Nguyen et al. (paper 2) — figure comparisons

**Primary replication tracker** for the DSQN / closed-loop neuromorphic DBS paper. Spec: [controllers/snn/replication.md](../controllers/snn/replication.md). Work proceeds **in parallel** with Mehregan ([paper_1.md](paper_1.md)); schedule by **panel**, not by waiting for paper 1 to fully close.

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/papers/2/`; JSON caches to `artifacts/figures/papers/2/`.

Figs **1–2** are schematics (CBGT diagram, closed-loop block diagram) — not replication targets.

| Panel | Script | Spec | Status |
|-------|--------|------|--------|
| Fig 3 — GPi α–β distribution (PD vs healthy) | `scripts/figures/papers/2/3/plot.py` | [snn/replication.md](../controllers/snn/replication.md) §3 | Smoke: ordering pass; θ-scale open |
| Fig 4 — training reward + episode length | `scripts/figures/papers/2/4/plot.py` (planned) | [snn/replication.md](../controllers/snn/replication.md) §6–§8 | Open |
| Fig 5 — CBGT spikes + DBS energy over training | `scripts/figures/papers/2/5/plot.py` (planned) | [snn/replication.md](../controllers/snn/replication.md) §8 | Open |
| Fig 6 — α–β + DBS params over training | `scripts/figures/papers/2/6/plot.py` (planned) | [snn/replication.md](../controllers/snn/replication.md) §8 | Open |
| Fig 7 — 50-episode eval (25 steps) | `scripts/figures/papers/2/7/plot.py` (planned) | [snn/replication.md](../controllers/snn/replication.md) §8 | Open |

Replication PNGs: `figures/papers/2/`. JSON caches: `artifacts/figures/papers/2/`.

---

## Fig 3 — GPi α–β oscillation power distribution

Distribution of GPi **α–β** power (**7–35 Hz**) for **PD** vs **healthy** (no DBS). Paper chooses control threshold **θ = 150** near the PD first quartile.

### Qualitative gates

- **Ordering:** PD samples sit **above** healthy on α–β power (median / bulk of mass).
- **Threshold:** θ = 150 is a plausible lower-quartile cut on the PD sample (not a hard CI number until sample size is locked).
- **Axes:** boxplot or sample scatter matching the paper’s two-panel idea (samples + summary).

### Status

**Smoke (short seeds):** ordering gate **pass** (PD median above healthy). Soft θ≈PD Q1 gate **open** — short 1 s / few-seed samples sit near ~500 on our α–β index while paper θ = 150; treat scale calibration as follow-up, not a blocker for the train path.

Replication image: [figures/papers/2/3/alpha_beta_dist.png](../../figures/papers/2/3/alpha_beta_dist.png)

**Run:**

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/2/3/plot.py
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/2/3/plot.py --seeds 0,1,2 --duration-s 1.0
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/2/3/plot.py --plot-only
```

---

## Fig 4 — Training rewards and lengths

Rewards and episode lengths over **500** training episodes. Qualitative: strong exploration early; shift toward exploitation around episode **~100**.

### Qualitative gates

- Early episodes: high variance (exploration).
- Later episodes: reward trend improves / length shortens when early-termination on α–β &lt; 150 engages.
- Not required: pixel match of paper wiggles (seed-dependent).

### Status

Open — `DSQNTrainer` + `rl-dbs train --controller snn` wired; needs a real plant train + plot script.

---

## Fig 5 — Network spikes and DBS energy

Per-episode CBGT spike counts and approximate DBS energy (Eq. (6)) over training.

### Qualitative gates

- Energy trace responds as DBS parameters move.
- Spike / energy panels are readable at paper scale (ordering and rough shape, not digit match).

### Status

Open.

---

## Fig 6 — α–β and DBS parameters over training

GPi α–β and the three DBS parameters (amplitude, frequency, pulse width) over 500 episodes. Paper end-of-training anchors (~262 nA/cm², ~78.65 Hz, ~1 ms) are **evaluation anchors**, not hard gates until one-seed qualitative shape passes.

### Qualitative gates

- α–β decreases relative to PD baseline / toward or below θ = 150.
- Parameters leave the init triple (40 Hz, 0.3 ms, 300 nA/cm²) and settle in a plausible therapeutic band.
- Exact paper readouts optional after shape passes.

### Status

Open.

---

## Fig 7 — Evaluation (50 episodes × 25 steps)

Seeded eval rollouts of the trained policy.

### Qualitative gates

- Mean α–β under the learned policy below untreated PD reference from Fig 3.
- Protocol: **50** episodes, **25** steps, varying seeds.

### Status

Open — needs train + `rl-dbs eval --controller snn`.

---

## Conventions (v1)

Documented choices for paper-silent knobs (see [snn/replication.md](../controllers/snn/replication.md) §12 and `SNNConfig`):

| Knob | v1 choice |
|------|-----------|
| Observation | GPi-only binary spike matrix |
| Action head | `factored` (3× ternary argmax) |
| Early-stop streak $t_u$ | 3 |
| LIF θ_th | 1.0 |
| Target net | hard copy on a fixed period |
| Reward coeffs δ, τ | `energy_penalty=0.01`, `threshold_reward=1.0` |
