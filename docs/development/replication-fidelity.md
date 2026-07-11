# Replication fidelity — Mehregan et al. (adaptive DBS / quantization)

Single source of truth for **what we have replicated**, **what diverges**, and **what we added** when reproducing Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*. This document consolidates findings from ~150 tasks; detailed specs remain in [environment.md](../environment.md), [plant.md](../plant.md), and [controllers/ddpg/replication.md](../controllers/ddpg/replication.md).

**How to use:** Before training, benchmarking, or claiming paper alignment, check the tables below. When a row says **verified**, cite the evidence column. When a row says **deviation** or **hypothesis**, do not treat it as paper-grounded without further validation.

**Other controllers:** Nguyen (SNN) and Ravivarapu (SEA-DBS) have separate replication specs under `docs/controllers/snn/` and `docs/controllers/sea_dbs/`. This document covers **Mehregan DDPG** only.

---

## 1. What matches the paper (verified)

| Area | Paper / spec | Our implementation | Evidence |
|------|--------------|-------------------|----------|
| **Plant model** | Kumaravelu et al. (2016) CBGT, parkinsonian (`pd=1`), STN DBS | Bundled MATLAB reference + native `PythonPlant` port | TASK-106; `tests/envs/plant_backend_equivalence_test.py`; [plant.md](../plant.md) §8 |
| **Python vs MATLAB plant** | Same dynamics on shared grid | GPi spikes and $P_\beta$ match within ~1e-15 spike-time atol / &lt;1% $P_\beta$ rel error (init fixtures) | TASK-106; `tests/envs/python_integrator_fixed_ic_test.py` |
| **Beta biomarker** | Eq. (1): GPi PSD integral **13–35 Hz**, mean over $n=10$ neurons | `envs/plant/biomarkers.py` — 13–35 Hz (not Kumaravelu default 7–35) | [environment.md](../environment.md) §3.1; TASK-109 |
| **Fig. 1b (GPi PSD)** | Healthy, PD, PD + 130 Hz cDBS GPi PSD panel | `scripts/figures/papers/1/1b/plot.py`; seeds 0–9 mean, 10 s segment; qualitative $P_\beta$ ordering matches paper | 2026-07-09; [plant.md](../plant.md) §8.1; [paper_1.md](../figures/paper_1.md) |
| **Fig. 2a (GPi $P_\beta$ time series)** | PD no treatment vs PD + 130 Hz cDBS; 12 s, 2 s onset at $t=2$ | `scripts/figures/papers/1/2a/plot.py`; seed 0 qualitative match; 130 Hz suppresses beta after onset | 2026-07-09; [plant.md](../plant.md) §8.2; [paper_1.md](../figures/paper_1.md) |
| **Reward Eq. (8)** | $\beta_t = 0.35$; positive below threshold; quadratic penalty at/above; **no energy term** | `MehreganEnvConfig.beta_threshold=0.35`; linear branch negated to match Fig. 3c prose (TASK-78) | TASK-109; [environment.md](../environment.md) §6 |
| **DBS amplitude** | 300 nA/cm² | `DbsSpec.amplitude` default | [plant.md](../plant.md) §4 |
| **Pulse width** | 0.3 ms | `DbsSpec.pulse_width_ms=0.3` | [plant.md](../plant.md) §4 |
| **RL step duration** | 2 s simulated time per transition | `step_duration_s=2.0` | [environment.md](../environment.md) §5 |
| **Episode length** | 30 steps per episode | `max_episode_steps=30` | [environment.md](../environment.md) §5 |
| **Training hyperparameters (§IV.A.1)** | actor_lr $5\times10^{-4}$, critic_lr $10^{-3}$, buffer 8192, batch 32 | `DDPGConfig` defaults | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) §4 |
| **Reported training budget** | 10 episodes | `num_episodes=10` default | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) §4.4 |
| **State / observation** | Within-step biomarker series over full 2 s segment (interpretation II) | `state_mode=within_step`, `state_length=16` default; $L$ sub-window $P_\beta$ / `observation_scale` | TASK-158; [environment.md](../environment.md) §3.2 |
| **Reward Eq. (8) input** | Biomarker window = segment length (§IV.A.1) | `reward_state_mode=full_segment` — whole-step $P_\beta$ for $s_{\mathrm{sum}}$; CNN obs stays within-step | TASK-162; [environment.md](../environment.md) §3.2 |
| **Actor CNN (Fig. 3a)** | Conv1→32→AvgPool→Conv→64→AvgPool→Linear(256)→Linear(256)→logits | `controllers/ddpg/networks.py` | TASK-146 |
| **Critic fusion (Fig. 3b)** | State trunk + action branch; **element-wise add** | `CriticNetwork` add fusion | TASK-146 |
| **Action selection** | Discrete patterns; softmax + argmax at deploy | `select_action(logits)` | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) §3.2 |
| **Init mean frequency** | Regular pulses at **45 Hz** (main); **30 Hz** ablation | Variants `paper` / `init-30hz`; pattern mode `pattern_mean_hz` | [benchmarking.md](../benchmarking.md) |
| **Pattern semantics (Option C)** | Fixed mean rate; agent shapes temporal pattern, not scalar frequency | `FixedMeanPatternAlphabet` — 41 patterns at fixed `mean_hz` | TASK-83/84; [environment.md](../environment.md) §4.2 |
| **Pattern landscape (Eq. 8)** | At 30 Hz init, irregular &gt; periodic (Fig. 5b) | `FixedMeanPatternAlphabet` 1-step + 30-step probes | TASK-156: at 30 Hz (`dt_ms=0.02`), pattern 0 worst (41/41); best irregular action 19 (1-step) / mid-irregular action 10 (30-step total −106 vs pattern 0 −278). At 45 Hz pattern 0 remains best open-loop. |
| **Fig 5b gate (30 Hz)** | Trained policy Pβ &lt; no-stim and &lt; periodic 30 Hz | TASK-166/170 fail: trained ≈542–545 &gt; no-stim ≈503 | TASK-172: at `dt_ms=0.02` **zero** open-loop patterns beat no-stim (best Pβ≈520); collapsed policies pick suboptimal actions 8/30. See [task172-fig5b-paper-gap-analysis.md](task172-fig5b-paper-gap-analysis.md). |
| **PTQ / QAT hooks** | §III.D quantization experiments | `controllers/ddpg/quantization.py`; benchmark slugs `ptq-fp16`, `ptq-int8`, `qat` | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) §6 |

---

## 2. What does not match or remains uncertain

### 2.1 Documented deviations (we know we differ)

| Topic | Paper | Ours | Impact | Mitigation |
|-------|-------|------|--------|------------|
| **Plant $\Delta t$** | §IV.A.1 reports **0.02 ms** | `PlantConfig.dt_ms` defaults to **0.01 ms** (Kumaravelu reference) | Finer integration grid; biomarker stats may differ slightly | Set `plant.dt_ms: 0.02` in `.rl-dbs.yaml` for paper-aligned runs; document in results |
| **Default action space** | Discrete **pulse patterns** at fixed mean rate | `action_space_mode` defaults to **`scalar_frequency`** (0:5:200 Hz grid) for backward-compatible benchmarks | Scalar mode makes **constant frequency** the cost-free optimum — cannot express paper's irregular patterns | Use `fixed_mean_pattern` for faithful Mehregan replication (TASK-83) |
| **Training exploration** | Paper does not specify online exploration; deploy uses greedy argmax | **ε-greedy** 0.5→0.1 (default) or softmax temperature anneal | Extra variance in replay; critic sees off-policy actions | Paper-faithful runs: greedy argmax (`exploration_mode` / ε=0); `critic_action_input=one_hot` when exploring |
| **Critic action input under exploration** | Fig. 3b / Algorithm 1 tuple uses **actor logits** | Default **`one_hot`** for executed discrete action | Correct Q targets when ε-greedy overrides argmax; differs from paper diagram under exploration | Use `logits` only when interaction is fully greedy |
| **Episode count in experiments** | §IV.A.1 reports **10** episodes | Various diagnostic runs used 20+ episodes | Longer runs may change learning curves | Label non-10-episode runs as extensions; use `num_episodes=10` for paper comparison |
| **30 Hz focus** | Paper trains **both** 45 Hz and 30 Hz init | Adaptive validation has emphasized **30 Hz** pattern mode | 45 Hz paper-scale benchmark exists (`paper` variant) but pattern-mode learning is newer | Run both `paper` and `init-30hz` / `pattern_mean_hz=30` when claiming full replication |
| **QAT in pattern mode** | §IV.A.3 QAT with 10 episodes | QAT benchmarked on **scalar-frequency** checkpoints only | Pattern-mode QAT not yet validated | Treat as open until TASK-70+ pattern re-benchmark |

### 2.2 Paper-silent choices (implementation hypotheses)

These are **deliberate conventions**, not verified against released code (none available per [AGENTS.md](../../AGENTS.md)):

| Topic | Our choice | Rationale | Risk |
|-------|------------|-----------|------|
| **Pattern alphabet size** | **41** patterns | Matches scalar head width / Kumaravelu freq grid size | Paper may use a different cardinality |
| **Irregular pattern construction** | ±**1/3 ISI** jitter; first/last onsets pinned; seeded by `(mean_hz, index)` | Preserves mean rate and span; pulses stay separated | Mehregan's learned patterns may lie **outside** this alphabet (TASK-109) |
| **Observation scale** | $s = P_\beta / 1000$ | Maps unstimulated $P_\beta$ ~400–500 to $s \approx 0.4$–$0.5$, aligning with $\beta_t = 0.35$ | Paper never defines the mapping; TASK-164: scale 2000 re-collapses 30 Hz landscape to positive branch |
| **$\gamma$, $\tau$, update frequency** | 0.99, 0.005, 1 step/update | Standard DDPG defaults; §IV.A.1 silent | May differ from authors' unreleased code |
| **Eval segment timing** | 10 s protocol ambiguous ($2 + 5 \times 2\text{s} > 10\text{s}$) | Leading hypothesis: **1.6 s** per eval segment | Cross-paper metric comparison may shift |

### 2.3 Known bugs fixed (historical — do not regress)

| Bug | Symptom | Fix | Task |
|-----|---------|-----|------|
| Reward sign (below $\beta_t$) | Agent penalized for suppressing beta | Negate linear branch: $R = (\beta_t - s_{\mathrm{sum}}) \cdot 10$ | TASK-67/78 |
| Scalar-frequency action space | Constant policy "collapse" looked like a bug | Root cause: wrong action semantics vs paper; use pattern mode | TASK-81 |
| `state_length=15` + `multi_step_history` | CNN sees stale cross-step history; inevitable constant policy | Paper path uses `within_step` series; extension labeled | TASK-67 / TASK-158 |

---

## 3. What we added beyond the paper (extensions)

Repo features that are **intentionally not** Mehregan replication. Disable or document when running paper-faithful protocols.

| Extension | Purpose | Default in paper-faithful runs |
|-----------|---------|-------------------------------|
| **v2 exploration stack** (`logit_noise_std`, `entropy_coeff`, `random_warmup_steps`, `critic_warmup_steps`, temperature decay) | Debug constant-policy collapse | **Off** — use greedy or basic ε-greedy only |
| **`multi_step_history`** | Rolling deque of past whole-step scalars (TASK-67 investigation) | **Off** — use `within_step` for paper path |
| **`state_length=1` shortcut** | Scalar $P_\beta$ per step (pools skipped) | Optional for fast probes; default paper path is `within_step` + `L≥4` |
| **Scalar frequency action space** | Phase 1–4 benchmarks, cDBS baselines | **Off** for pattern replication |
| **`obs_normalize` / running stats** | Experimental obs preprocessing | **Off** |
| **`init_bias_scale` tuning** | Faster convergence experiments (e.g. learning_v1) | Paper default **2.0** for benchmark checkpoints |
| **Softmax exploration mode** | Alternative to ε-greedy | Not paper-specified |
| **Longer training / custom episode scripts** | `run_pattern_train*.py`, resilient trainers | Label as extension; compare at 10 episodes separately |
| **Native Python plant + Numba** | ~10× speed vs MATLAB | Same dynamics when init fixtures match; default backend `python` since TASK-63 |

---

## 4. Open questions

Questions that block **qualitative** claims about matching paper figures. Track answers in task comments; update this section when resolved.

| # | Question | Why it matters | Status |
|---|----------|----------------|--------|
| 1 | Does "swiftly achieves high rewards" mean convergence within **10 episodes** at **30 Hz** init? | Defines success criterion for adaptive validation | **Root cause found (TASK-160/162)** — coupled `observation_mean` reward collapsed landscape; fix: `reward_state_mode=full_segment`. TASK-159 fail predates fix; retrain gated on smoke (TASK-163). |
| 2 | Exact STN **waveform** (rectangular on/off vs charge-balanced / biphasic)? | Plant drive semantics | **Open** — we use Kumaravelu rectangular pulses |
| 3 | Does the paper use **charge-balanced** waveforms? | Energy delivery fidelity | **Open** — Kumaravelu reference is monophasic rectangular |
| 4 | How are Mehregan's **discrete patterns** constructed (alphabet, jitter, boundaries)? | Pattern-mode hypothesis validation | **Open** — our ±1/3 ISI / 41-pattern scheme is documented hypothesis |
| 5 | Post-training **eval segment duration** within the 10 s budget | Benchmark comparability | **Open** — 1.6 s/segment hypothesis in [environment.md](../environment.md) §8 |
| 6 | Does **pattern 0** (regular train) beat irregular patterns on the paper's landscape? | Fig. 5b irregular &gt; periodic at 30 Hz | **Settled (TASK-156, Branch A)** — at **30 Hz** with `plant.dt_ms=0.02`, pattern 0 ranks **41/41** (1-step) and loses every 30-step constant-policy rollout; **≥1 irregular clearly beats pattern 0** on both probes. At **45 Hz**, pattern 0 still ranks **1/41** (1-step) and wins 30-step rollouts — open-loop landscape is mean-rate dependent, not a global contradiction of Fig. 5b (which is 30 Hz specific). Evidence: `artifacts/ddpg/q6_landscape_*.json`. **DDPG retrain still gated** — landscape shows irregular *can* win at 30 Hz, but trained-policy acceptance criteria unchanged; COO must greenlight before TASK-70/83. |

---

## 5. Validation status by experiment type

| Experiment | Action space | Plant backend | Benchmark status | Notes |
|------------|--------------|---------------|------------------|-------|
| Phase 4 `mehregan_eval` | Scalar frequency | MATLAB (legacy) | **Complete** (TASK-9/94) | See [phase4-results.md](phase4-results.md); DDPG converged to ~112.5 Hz constant policy (expected for scalar mode) |
| Pattern-mode training | `fixed_mean_pattern` | PythonPlant | **TASK-159 FAIL** | `within_step` L=16, greedy+logits, 30 Hz, dt=0.02, seed 0 — **Fig 4a FAIL** (β_norm flat: 0.245→0.246), **Fig 4b FAIL** (ep1≈ep10 ~32.5; best ep4=36.6), **Fig 5b FAIL** (trained Pβ 574.8 &gt; no-stim 503.4; beats periodic 648.4). Constant policy action 20. Artifacts: `pattern_train_30hz_L16_fig45_*`. TASK-70 remains blocked. |
| Paper-protocol eval | Either | PythonPlant | Partial (TASK-108) | 10 s eval harness exists |
| QAT / PTQ | Scalar (checkpoints) | MATLAB | **Complete** for scalar suite | QAT underperforms FP on mean $P_\beta$ |
| Plant parity | N/A | Python vs MATLAB | **Verified** | TASK-106 |

---

## 6. Quick checklist — paper-faithful Mehregan DDPG run

Use this before claiming replication:

- [ ] `action_space_mode = fixed_mean_pattern`
- [ ] `pattern_mean_hz` = 45 (`paper`) or 30 (`init-30hz`)
- [ ] `state_length = 1`
- [ ] `num_episodes = 10`, `max_episode_steps = 30`
- [ ] `plant.dt_ms = 0.02` (or document 0.01 ms deviation)
- [ ] Greedy deploy / paper-faithful training exploration documented
- [ ] Reward Eq. (8) with corrected below-threshold sign
- [ ] CNN topology per TASK-146
- [ ] Compare against baselines: none, cDBS 130 Hz, periodic at init mean Hz

---

## 7. Related tasks (index)

| Task | Topic |
|------|-------|
| TASK-106 | PythonPlant ↔ MATLAB verification |
| TASK-109 | Reward audit; pattern alphabet uncertainty |
| TASK-67/78 | Reward sign; `state_length=1` |
| TASK-81 | Scalar vs pattern action space root cause |
| TASK-83/84 | Fixed-mean pattern alphabet (Option C) |
| TASK-105 | Pattern 0 semantics; reward landscape sweep (45 Hz default) |
| TASK-156 | Q6 settlement — 30/45 Hz landscape vs Fig. 5b |
| TASK-146 | CNN / critic architecture alignment |
| TASK-9/94 | Phase 4 scalar-frequency benchmark |
| TASK-70 | Pattern-mode re-benchmark (deferred) |
| TASK-159 | Bounded 30 Hz L=16 greedy+logits retrain — Fig 4/5b acceptance (**fail**) |

---

## 8. Maintenance

When replication scope changes:

1. Update the relevant spec (`environment.md`, `plant.md`, `controllers/ddpg/replication.md`) **and** this summary in the same pass.
2. Add a row to §1 (verified), §2 (deviation/hypothesis), or §3 (extension) — do not bury decisions only in code or task threads.
3. Link the validating task or test path in the **Evidence** column.
