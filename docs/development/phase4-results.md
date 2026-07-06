# Phase 4 results — Mehregan DDPG benchmarking

Status snapshot for Phase 4 exit criteria ([roadmap.md](roadmap.md) §2). **Implementation checklist:** verified in TASK-10 (2026-07-01). **Benchmark numbers:** full `mehregan_eval` suite complete (TASK-9, 2026-07-03).

---

## 1. Benchmark outcomes

Full suite: `suites/mehregan_eval.yaml` — 8 controllers × 5 eval seeds (40 planned runs) plus `baseline:periodic-30hz` (5 seeds, TASK-62, 2026-07-05). Train seed fixed at 0. **Delivered:** 46 run directories in `results/mehregan_eval/runs/` (one duplicate `ddpg:qat` seed 2 retained; metrics deduped by seed for tables below). Manifest `completed_runs` field is stale (shows 1); per-run `metrics.json` files are authoritative.

| Artifact | Path | Status |
|----------|------|--------|
| Raw run logs | `results/mehregan_eval/` | **Done** (41 runs) |
| Summary tables | `uv run rl-dbs summary --suite-name mehregan_eval` | **Done** — also `summary.csv` |
| Checkpoints | `artifacts/ddpg/<variant>_train0.pt` | **Done** — `paper`, `init-30hz`, `qat` |

**Variants:** `paper`, `init-30hz`, `ptq-fp16`, `ptq-int8`, `qat` (+ baselines `none`, `cdbs-130hz`, `periodic-45hz`, `periodic-30hz`).

### 1.1 Core metrics (mean over eval seeds, deduped)

| Controller | Variant | P_β mean | P_β final | Reward sum | Stim freq (Hz) |
|------------|---------|----------|-----------|------------|----------------|
| ddpg | paper | 191.8 | 138.5 | −11.83 | 112.5 |
| ddpg | init-30hz | 217.8 | 160.5 | −11.41 | 112.5 |
| ddpg | ptq-fp16 | 191.8 | 138.5 | −11.83 | 112.5 |
| ddpg | ptq-int8 | 191.8 | 138.5 | −11.83 | 112.5 |
| ddpg | qat | 205.8 | 167.5 | −10.82 | 108.8 |
| baseline | none | 458.2 | 458.2 | −7.50 | 0.0 |
| baseline | cdbs-130hz | 197.4 | 145.2 | −11.49 | 108.3 |
| baseline | periodic-45hz | 301.2 | 269.8 | −5.26 | 37.5 |
| baseline | periodic-30hz | 583.2 | 607.9 | −34.60 | 25.0 |

PTQ variants match full-precision `paper` on all reported metrics (expected — post-training quantize of the same checkpoint).

`periodic-30hz` is the Mehregan **30 Hz init ablation** baseline (not a therapeutic target — 30 Hz periodic STN drive **increases** $P_\beta$ vs `none` and `periodic-45hz`, as expected for sub-therapeutic frequency).

### 1.2 §IV qualitative checklist (per DDPG variant)

Per-seed pass counts from `results/mehregan_eval/` (same logic as `controllers/ddpg/checklist.py`; deduped by seed). Aggregate mean-level pass/fail in parentheses.

| Variant | lowers vs none | beats cdbs (5% slack) | mehregan_eval protocol | quantization_tagged |
|---------|----------------|----------------------|------------------------|---------------------|
| paper | 5/5 (pass) | 4/5 (pass mean) | 5/5 | n/a |
| init-30hz | 5/5 (pass) | 3/5 (fail mean) | 5/5 | n/a |
| ptq-fp16 | 5/5 (pass) | 4/5 (pass mean) | 5/5 | pass (matches FP) |
| ptq-int8 | 5/5 (pass) | 4/5 (pass mean) | 5/5 | pass (matches FP) |
| qat | 5/5 (pass) | 3/5 (borderline mean) | 5/5 | n/a |

### 1.3 Deviations from paper claims

Document any gaps between benchmark outcomes and Mehregan et al. §IV figures/tables once numbers land. Known **implementation conventions** (not necessarily deviations):

| Topic | Chosen convention | Spec reference |
|-------|-------------------|----------------|
| γ, τ, update frequency | 0.99, 0.005, 1 | [replication.md](../controllers/ddpg/replication.md) §9.3 |
| CNN topology | Conv1d 1→16→32, `shrink_dim=4`, adaptive pool | `controllers/ddpg/networks.py` |
| Eval segment | `eval_steps=5` after reset (≈10 s simulated at 2 s/step) | [environment.md](../environment.md) §8 |
| CDBS comparison slack | 5% on `p_beta_mean` vs `cdbs-130hz` | `controllers/ddpg/checklist.py` |
| QAT vs PTQ | Paper reports weaker QAT at 10 episodes; no automated threshold | checklist `paper_notes` only |

---

## 2. §8 consistency checklist audit (implementation)

Cross-check of [replication.md](../controllers/ddpg/replication.md) §8 against `controllers/ddpg/` (2026-07-01).

| §8 item | Verified | Evidence |
|---------|----------|----------|
| CNN actor over biomarker state; critic fuses state + logits | Yes | `networks.py` — `StateEncoder`, `Actor.head`, `Critic` concat |
| Discrete pattern via argmax; replay stores `a` and `a_logit` | Yes | `Actor.select_action`, `buffer.py` `Transition` |
| Bootstrap target with `(1-dw)` masking | Yes | `trainer.py` L119 |
| Critic MSE; actor maximizes Q with critic frozen | Yes | `trainer.py` L121–137 |
| Soft updates share τ for actor and critic targets | Yes | `trainer.py` L139–140, `config.tau` |
| LR 5e-4 / 1e-3; buffer 8192; batch 32; 10×30 steps; 2 s step | Yes | `config.py`, `envs/mehregan/config.py` `step_duration_s=2.0` |
| Init 45 Hz (`paper`) and 30 Hz ablation (`init-30hz`) | Yes | `init_baseline_for_variant`, `Actor.init_toward_action` |
| PTQ FP16/INT8 and QAT | Yes | `quantization.py`; tests in `quantization_test.py` |

**§6 quantization (inference paths):**

| Mode | Verified | Evidence |
|------|----------|----------|
| PTQ FP16 | Yes | `apply_ptq` → `actor.half()`; `test_ptq_fp16_eval` |
| PTQ INT8 | Yes | `quantize_dynamic` on Linear; `test_ptq_int8_prepare` |
| QAT fake-quant stubs | Yes | `QATActor` QuantStub/DeQuantStub; `test_qat_checkpoint_roundtrip` |

---

## 3. Tests

```bash
uv run pytest tests/controllers/ddpg/ -q -m "not matlab"
```

**Result (2026-07-01):** 26 passed, 1 deselected (`matlab_trainer_test` — requires MATLAB Engine on host).

| Area | Tests | Notes |
|------|-------|-------|
| Checklist | `checklist_test.py` | Pass/fail on synthetic summaries |
| Replication workflow | `replication_test.py` | Mock plant train→eval→baselines |
| Quantization | `quantization_test.py` | QAT train, PTQ eval, `mehregan_eval` protocol |
| Trainer / buffer / networks | `*_test.py` | Algorithm 1 unit coverage |

---

## 4. §IV qualitative checklist coverage (`checklist.py`)

`assess_replication_summary` implements:

| Check | Condition | Notes |
|-------|-----------|-------|
| `ddpg_lowers_p_beta_vs_none` | `p_beta_mean` (ddpg) < none | Core §IV claim |
| `ddpg_beats_or_matches_cdbs` | ddpg ≤ cdbs × 1.05 | 5% slack on mean beta power |
| `mehregan_eval_protocol` | `protocol == "mehregan_eval"` | Set by `run_mehregan_eval` / `evaluate` |
| `quantization_tagged` | `metrics_extra.quantization == variant` | **PTQ variants only** (`ptq-fp16`, `ptq-int8`) |

**Gaps (documented, not blockers for Phase 4 code audit):**

1. **QAT** — no `metrics_extra.quantization` tag in `evaluate()` (only `is_ptq_variant`); checklist skips quantization_tagged for `qat`.
2. **QAT qualitative threshold** — paper notes weaker suppression at 10 episodes; no automated pass/fail (by design in `paper_notes`).
3. **init-30hz baseline pairing** — enforced in `replication.py` / tests, not in `checklist.py` (checklist uses whatever baselines appear in summary JSON).

CLI: `uv run python scripts/check_mehregan_replication.py <summary.json>`

---

## 5. Phase 4 exit criteria status

| Criterion | Status |
|-----------|--------|
| Repeatable `mehregan_eval` across FP + quantized variants | **Done** — 40/40 planned evals + checkpoints (2026-07-03) |
| Replication checklist passable for `ddpg` | **Done** — code verified (TASK-10); plant-scale: all variants lower P_β vs none; `paper`/PTQ beat cdbs on mean; `init-30hz`/`qat` mixed on per-seed cdbs slack (documented §1.2) |
| `rl-dbs benchmark` / `rl-dbs-tui` usable | **Done** (see [benchmarking.md](../benchmarking.md)) |
| Fresh-VM setup scripts | **Windows Sandbox passed** (2026-06-30, `-Clone`); Multipass Linux pending — [fresh-validation.md](fresh-validation.md) |

---

## 6. Paper-variant policy investigation (TASK-37, 2026-07-03)

PTQ (`ptq-fp16`, `ptq-int8`) matching `paper` byte-for-byte on all benchmark metrics is **expected**, not a quantization bug.

### Root cause

The `paper` checkpoint (`artifacts/ddpg/paper_train0.pt`) implements a **state-collapsed constant policy**, not a softmax-uniform distribution over 41 actions:

| Check | Result |
|-------|--------|
| Argmax on biomarker-range obs (0.3–0.6, `state_length=1`) | **Always action 27** (135 Hz) across 5000 samples |
| Top-2 logit margin | mean **0.013** (stable; not softmax-uniform — entropy ≈ 3.47 vs max ln 41 ≈ 3.71) |
| `stim_frequency_mean` = 112.5 Hz | Matches `(0 + 5×135) / 6` — reset segment is 0 Hz, five eval steps at 135 Hz |
| PTQ argmax disagreements | **0 / 2000** (FP16 and INT8) on biomarker-range states |
| Init bias (45 Hz, action 9) | Still **1.94** (init `init_toward_action` = 2.0); encoder+weight row for action 27 dominates |

**Mechanism:** With `MehreganEnvConfig.state_length = 1`, the “temporal” CNN sees a single scalar $P_\beta/1000$. Conv1 → pool → Conv2 produces encoder features that vary only slightly across the operating biomarker range; the linear head then always selects action 27. Training did run (10 episodes; final block in `artifacts/ddpg/train_paper_seed0.log` stabilizes around reward −5.6) but converged to a **frequency-constant** policy near 135 Hz that still beats cDBS (130 Hz) on mean $P_\beta$.

**QAT / init-30hz differ for other reasons:**

| Variant | Policy shape | Benchmark $P_\beta$ mean |
|---------|--------------|--------------------------|
| `paper` / PTQ | Constant action 27 | 191.8 |
| `init-30hz` | Also constant action 27 (init bias overwritten) | 217.8 |
| `qat` | Mixed actions 27 / 16; lower margins (mean 0.007, ties at 0) | 205.8 |

QAT fake-quant stubs perturb logits enough to tie-break between actions, so plant trajectories diverge. `init-30hz` uses the same collapsed policy but a different training run (partial retrain visible in logs).

### Recommendations

1. **PTQ pass-through — no action needed.** Confirms post-training quantize preserves argmax on this checkpoint; checklist `quantization_tagged` pass is valid.
2. **Training / observation — partial (TASK-65):** default `state_length` raised to **15** (half-episode temporal window; best final reward in sweep). **All swept values still collapsed to constant policies** after 10 episodes (greedy argmax only). Next: add exploration and/or train longer; re-benchmark after retrain.
3. **Eval protocol — not the blocker.** Five eval steps is short, but the current policy ignores state; longer eval would not differentiate actions until the policy is state-dependent.

Evidence: offline analysis on `paper_train0.pt`, `init-30hz_train0.pt`, `qat_train0.pt` (2026-07-03); training log `artifacts/ddpg/train_paper_seed0.log`.

---

## 7. PythonPlant review + backend flip recommendation (TASK-62, 2026-07-05)

### Verification run

| Check | Result |
|-------|--------|
| `pytest tests/envs/python_plant_test.py python_integrator_fixed_ic_test.py plant_backend_equivalence_test.py -m "not matlab"` | **9 passed** |
| `pytest tests/envs/plant_backend_equivalence_test.py python_integrator_fixed_ic_test.py -m matlab` | **10 passed** (~9.5 min) |
| 2 s integrate speed (seed=42, no DBS, WSL) | **6.4 s** (≥10× vs MATLAB ≈58–66 s) |

### Implementation review (`envs/plant/`)

- **`PythonPlant`** (`python_backend.py`) — clean `PlantBackend` wrapper; delegates to `integrate_network` with exported init fixtures (`load_cached_init_draws`) for MATLAB `randperm` parity.
- **Integrator** (`network/integrator.py` + `numba_loop.py`) — Numba JIT path is default when available; NumPy fallback for debug/trace. Spike detection fix (upward crossing only) documented in [native-plant-port.md](native-plant-port.md).
- **Config surface** — default `plant.backend: python` (TASK-63); override with `matlab` or `RL_DBS_PLANT_BACKEND=matlab` when MATLAB bridge is needed.

### Backend flip recommendation

**Do not flip the default to `python` yet.**

| Gate | Status |
|------|--------|
| GPi / $P_\beta$ parity vs MATLAB | **Pass** |
| DBS ordering (130 Hz / 45 Hz) | **Pass** |
| ≥10× speedup (2 s integrate) | **Pass** (6.4 s vs ≈60 s) |
| Full `mehregan_eval` baseline on Python (`none`, `cdbs-130hz`, seeds 0–4) | **Pass** (TASK-63, 2026-07-06) — bit-exact `p_beta_mean` vs MATLAB; DBS ordering preserved; default backend flipped to `python` |

**TASK-63 results (2026-07-06):** Exported `plant_init_seed{0..4}.npz` fixtures; ran `RL_DBS_PLANT_BACKEND=python uv run rl-dbs eval` for `baseline:none` and `baseline:cdbs-130hz` (seeds 0–4). All 10 runs: relative error &lt; 1e-15 on `p_beta_mean` vs `results/mehregan_eval/summary.csv`; `cdbs-130hz` &lt; `none` on every seed. Python runs in `results/mehregan_eval_python/`. Wall time ~3 min per variant (5 seeds sequential).

---

## 8. state_length sweep (TASK-65, 2026-07-06)

Parallel short DDPG training (`paper` variant, seed 0, 10 episodes, PythonPlant) for `state_length ∈ {5, 10, 15, 30}`. Script: `scripts/state_length_sweep.py`; merged results: `artifacts/ddpg/state_length_sweep.json`.

| `state_length` | Final reward | Offline unique | Rollout unique | Dominant action (Hz) |
|----------------|-------------:|---------------:|---------------:|---------------------:|
| 5 | −56.74 | 1 | 1 | 37 (185) |
| 10 | −38.18 | 1 | 1 | 19 (95) |
| **15** | **−18.17** | 1 | 1 | 9 (45) |
| 30 | −27.23 | 1 | 1 | 27 (135) |

**Selection:** No value achieved `rollout_unique > 1` or `offline_unique > 1` (acceptance criterion unmet for adaptive policy). Among `state_length > 1`, **15** wins on `final_reward` tiebreaker and matches a reasonable temporal window (15 of 30 episode steps). Default updated in `envs/mehregan/config.py`.

**Implication:** Temporal observations alone do not break constant-policy collapse under 10-episode greedy DDPG. Parent TASK-64 PTQ/QAT retrain remains blocked until exploration or longer training yields state-dependent policies.

---

## 9. Follow-up

- [ ] Fix stale `manifest.json` `completed_runs` (cosmetic; `summary.csv` is current)
- [ ] Optional: tag `metrics_extra.quantization` for `qat` in `evaluate()` for symmetry with PTQ
- [ ] Optional: add checklist test for `ptq-int8` quantization_tagged path
- [ ] Multipass Linux fresh-validation (remaining Phase 4 portability item)
- [ ] **TASK-64 unblock:** retrain `paper` with `state_length=15`, exploration, and/or >10 episodes; verify `rollout_unique > 1`; re-run `mehregan_eval`

---

## 10. DDPG architecture / reward review (TASK-71, 2026-07-06)

**Trigger:** TASK-67 ε-greedy 10-ep run failed acceptance (`rollout_unique == offline_unique == 1`). TASK-67 30-ep softmax was still in flight when this review ran; findings below use ε-greedy + state_length sweep + offline probes.

### 10.1 Evidence summary

| Run | Episodes | Exploration | `rollout_unique` | Dominant action (Hz) | Final reward |
|-----|----------|-------------|------------------|----------------------|-------------:|
| state_length sweep (§8) | 10 | none (greedy) | 1 | varies by `state_length` | best −18.17 @ sl=15 |
| TASK-67 ε-greedy | 10 | 0.3→0.05 | 1 | 23 (115) | −36.38 |
| TASK-67 softmax 30-ep | 30 | 2.0→0.5 | *pending* | — | — |

Artifacts: `artifacts/ddpg/explore_sl15_seed0.json`, `artifacts/ddpg/state_length_sweep.json`, offline probe `artifacts/ddpg/task71_offline_probe.json`. Script: `scripts/ddpg_diagnose.py`.

### 10.2 Investigation findings

#### 1. Actor–critic gradient flow

- **Encoder is not dead.** On 500 synthetic windows (normalized $P_\beta/1000 \in [0.25, 0.65]$), encoder feature std is **0.22–0.29** and range **0.86–1.27** for `paper_train0.pt` and `paper_train0_explore.pt` — features vary across biomarker windows.
- **Policy head still collapses.** Despite varying encoder outputs, **offline argmax is unique=1** for both checkpoints. Explore training *increased* logit margin (mean **1.47** vs **0.013** for original `paper`) — the actor became *more* decisive, not more state-dependent.
- **Critic checkpoints are not saved**, so post-hoc $Q(s,a)$ variance across actions could not be measured on trained critics. Actor loss is $-\mathbb{E}[Q(s, \pi(s))]$; if the critic assigns similar $Q$ across logits for a given $s$, the actor receives no pressure to differentiate actions. **Likely bottleneck:** critic does not learn action-discriminative values, not encoder saturation.
- **Replay stores behavior logits** while actor update uses fresh $\pi(s)$ logits — standard for this paper-aligned formulation, but means critic must generalize across logit vectors, not discrete action indices.

#### 2. CNN capacity

- Current topology: Conv1d $1\to16\to32$, `AdaptiveAvgPool1d(shrink_dim=4)`, linear head → 41 actions (`networks.py`).
- Encoder features already vary; collapse happens at the **linear head** (single dominant logit row). Capacity increase is a reasonable next experiment but **not the first-order fix** — try weaker init + longer training + stronger exploration before a architecture sweep.

#### 3. Reward signal / normalization

- **Normalization is internally consistent:** observation $s(i) = P_\beta / 1000$; reward Eq. (8) uses `beta_threshold=0.35`, `reward_scale=10` (`envs/mehregan/reward.py`, `environment.md` §6).
- Synthetic reward range across $s_{\mathrm{sum}} \in [0.2, 0.6]$ is **6.25** (max **0** at $s_{\mathrm{sum}}=0.35$; all rewards $\leq 0$). The signal is non-flat in biomarker space.
- **Constant-frequency policies can still “win”.** Benchmark §1.1: constant **115 Hz** (action 27) lowers mean $P_\beta$ vs `none` and beats cDBS on mean — so RL can converge to a **frequency-constant** local optimum without state dependence. The reward does not penalize stimulation energy; there is no incentive to modulate frequency once a good constant is found.

#### 4. Actor initialization

- `init_toward_action` zeros all weights and sets `bias[init_action]=2.0` (45 Hz → action 9). After training, **head bias argmax remains action 9** but **argmax policy is action 23** (explore) or **27** (original `paper`) — weights moved, but to another **single** action.
- Strong init + decisive head can anchor early replay; weaker init reduces risk of premature commitment.

#### 5. Training budget

- PythonPlant: **~4 min/episode** (30 steps × 2 s integrate). 10 episodes ≈ 40 min; 30 ≈ 2 h. TASK-67 30-ep ε-greedy was killed after ~2 h with no log output (CPU contention). Budget is a practical constraint, not evidence that 30 ep would suffice — paper reports learning in 10 ep with exploration we do not yet match.

### 10.3 Root-cause synthesis

| Factor | Verdict |
|--------|---------|
| Missing exploration | **Necessary but not sufficient** — ε-greedy 10-ep still collapsed |
| `state_length` | **Helpful for reward** (§8) but alone does not break collapse |
| Reward / normalization bug | **Unlikely** — mapping matches spec |
| CNN encoder dead | **Ruled out** — features vary; head does not |
| Constant-policy local optimum | **Primary hypothesis** — reward + plant allow good constant Hz; critic does not force state–action coupling |
| Init + short training | **Contributing** — strong bias, 10 ep, decisive margins after explore |

### 10.4 Proposed config changes (experiment profile `learning_v1`)

Apply together on `state_length=15`, PythonPlant, seed 0. Tune one knob at a time if debugging.

```python
DDPGConfig(
    variant="paper",
    num_episodes=50,              # was 10; ~3.5 h on PythonPlant
    exploration_mode="softmax",   # TASK-67 softmax 30-ep result pending
    exploration_temperature_start=3.0,
    exploration_temperature_end=1.0,  # slower anneal than 2.0→0.5
    exploration_epsilon_start=0.9,    # if using epsilon instead
    exploration_epsilon_end=0.2,
    conv_channels=32,             # was 16
    shrink_dim=8,                 # was 4
    # init: reduce init_toward_action bias from 2.0 → 0.5 (code change in networks.py)
)
```

**Acceptance (unchanged from TASK-67):** `rollout_unique > 1` AND `offline_unique > 1` on `scripts/run_explore_retrain.py` / `scripts/ddpg_diagnose.py` probes.

**If `learning_v1` still fails:**

1. Save critic in checkpoints and log per-minibatch $\mathrm{std}_a Q(s, a)$ during training.
2. Try discrete-action critic input (one-hot action index) instead of full logit vector — larger spec deviation, document in `replication.md`.
3. Add mild frequency penalty to reward (extension, not paper replication) to break constant-Hz optimum.

### 10.5 Follow-up tasks

- [x] `init_bias_scale` on `DDPGConfig` / `Actor.init_toward_action` (default 2.0; `learning_v1` uses 0.5).
- [ ] Complete TASK-67 softmax 30-ep; append row to §10.1 table.
- [ ] Run `learning_v1` retrain when CPU budget available; re-benchmark if acceptance passes.
