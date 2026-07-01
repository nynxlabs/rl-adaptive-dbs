# Phase 4 results — Mehregan DDPG benchmarking

Status snapshot for Phase 4 exit criteria ([roadmap.md](roadmap.md) §2). **Implementation checklist:** verified in TASK-10 (2026-07-01). **Benchmark numbers:** pending full `mehregan_eval` suite run (TASK-9).

---

## 1. Benchmark outcomes (TBD — TASK-9)

Full suite: `suites/mehregan_eval.yaml` — 8 controllers × 5 eval seeds (40 runs). Train seed fixed at 0.

| Artifact | Path | Status |
|----------|------|--------|
| Raw run logs | `results/mehregan_eval/` | **Pending** — directory not yet populated |
| Summary tables | `uv run rl-dbs summary --suite-name mehregan_eval` | **Pending** |
| Checkpoints | `artifacts/ddpg/<variant>_train0.pt` | **Pending** |

**Variants:** `paper`, `init-30hz`, `ptq-fp16`, `ptq-int8`, `qat` (+ baselines `none`, `cdbs-130hz`, `periodic-45hz`).

When TASK-9 completes, fill §1.1–§1.3 below from `rl-dbs summary` output and per-variant checklist reports (`scripts/check_mehregan_replication.py` on replication summaries).

### 1.1 Core metrics (mean over eval seeds)

| Controller | Variant | P_β mean | Reward sum | Stim freq (Hz) | Checklist |
|------------|---------|----------|------------|----------------|-----------|
| ddpg | paper | TBD | TBD | TBD | TBD |
| ddpg | init-30hz | TBD | TBD | TBD | TBD |
| ddpg | ptq-fp16 | TBD | TBD | TBD | TBD |
| ddpg | ptq-int8 | TBD | TBD | TBD | TBD |
| ddpg | qat | TBD | TBD | TBD | TBD |
| baseline | none | TBD | TBD | — | — |
| baseline | cdbs-130hz | TBD | TBD | — | — |
| baseline | periodic-45hz | TBD | TBD | — | — |

### 1.2 §IV qualitative checklist (per DDPG variant)

Automated via `controllers/ddpg/checklist.py` → `assess_replication_summary`. Run after each variant's replication summary JSON is available.

| Variant | ddpg_lowers_p_beta_vs_none | ddpg_beats_or_matches_cdbs | mehregan_eval_protocol | quantization_tagged |
|---------|----------------------------|----------------------------|------------------------|---------------------|
| paper | TBD | TBD | TBD | n/a |
| init-30hz | TBD | TBD | TBD | n/a |
| ptq-fp16 | TBD | TBD | TBD | TBD |
| ptq-int8 | TBD | TBD | TBD | TBD |
| qat | TBD | TBD | TBD | n/a |

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
| Repeatable `mehregan_eval` across FP + quantized variants | **Infra done**; full-suite numbers **pending TASK-9** |
| Replication checklist passable for `ddpg` | **Code + unit tests verified**; plant-scale pass/fail **pending TASK-9** |
| `rl-dbs benchmark` / `rl-dbs-tui` usable | **Done** (see [benchmarking.md](../benchmarking.md)) |
| Fresh-VM setup scripts | **Windows Sandbox passed** (2026-06-30, `-Clone`); Multipass Linux pending — [fresh-validation.md](fresh-validation.md) |

---

## 6. Follow-up

- [ ] TASK-9: populate §1 tables from `results/mehregan_eval/` and update checklist columns
- [ ] Optional: tag `metrics_extra.quantization` for `qat` in `evaluate()` for symmetry with PTQ
- [ ] Optional: add checklist test for `ptq-int8` quantization_tagged path
