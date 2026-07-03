# Native Python plant port — plan and spec

**Status:** In progress (Phase B–D; see [roadmap.md](roadmap.md)).  
**Parent task:** Parallel MATLAB engines + native plant planning (TASK-15).  
**Authoritative plant spec:** [plant.md](../plant.md). Reference dynamics: `reference-material/KumaraveluEtAl2016/simulate_network_model.m`.

---

## 1. Motivation

MATLAB Engine integration is the training bottleneck. Each 2 s RL step integrates the CBGT network at **dt = 0.01 ms** over **200,000** micro-steps (~55–65 s per step on current hardware). A single DDPG training run (hundreds of episodes × tens of steps) is impractical for iteration.

| Approach | Expected speedup | Effort | When |
|----------|------------------|--------|------|
| **Parallel MATLAB engines** (TASK-15 child) | ~N× on independent seeds (N ≈ 3–4 on 17 GB RAM) | Small (days) | **Immediate** |
| **Native Python plant** (this doc) | 10–100× per step (NumPy baseline; higher with JAX/CuPy) | Large (weeks) | **Next** — unblocks practical training |

Parallel MATLAB is a bridge; native Python is the long-term fix. Both are in scope for TASK-15.

---

## 2. Goals and non-goals

### Goals

1. **`PythonPlant`** backend implementing the existing `PlantBackend` protocol ([`envs/mehregan/env.py`](../../envs/mehregan/env.py)) with the same `integrate()` / `reset()` / `close()` contract as `MatlabPlant`.
2. **Bit-for-bit or documented-tolerance equivalence** to the Kumaravelu MATLAB reference for:
   - GPi spike trains (fixed seed, `pd = 1`, no DBS)
   - $P_\beta$ (13–35 Hz, already computed in Python — [`envs/plant/biomarkers.py`](../../envs/plant/biomarkers.py))
   - DBS frequency response (e.g. 130 Hz, 45 Hz pattern indices)
3. **Drop-in replacement** — `MehreganEnv`, benchmark runner, and DDPG trainer accept `plant=` without other code changes.
4. **Reproducible validation** — extend existing `@pytest.mark.matlab` equivalence suite to parametrize backend (`matlab` vs `python`).

### Non-goals (initial port)

- Changing network topology, neuron count ($n = 10$), or default $\Delta t$.
- Porting Chronux/multitaper code (MATLAB `make_Spectrum` / `mtspectrumpt`) — biomarkers stay in Python.
- Mehregan pattern alphabet or reward logic (already in `envs/mehregan/`).
- GPU-first rewrite (optional follow-up after CPU equivalence).
- Julia port (deferred unless Python profiling shows insufficient gain).

---

## 3. Reference decomposition

`simulate_network_model.m` (~1,280 lines) breaks down as:

| Component | Lines (approx.) | Port? | Notes |
|-----------|-----------------|-------|-------|
| `simulate_network_model` entry | 1–89 | Partial | Orchestration only; Python `PythonPlant.integrate` replaces |
| `creatdbs` | 91–108 | **Yes** | STN pulse train from frequency; mirror [`envs/plant/dbs.py`](../../envs/plant/dbs.py) |
| `CTX_BG_TH_network` | 112–839 | **Yes** | Core integrator — **largest piece** |
| `find_spike_times` / `spikes_to_cell` | 840–858 | **Yes** | Partially exists in [`envs/plant/spikes.py`](../../envs/plant/spikes.py) |
| GPe / TH / STN gating functions | 860–1036 | **Yes** | Direct translation to NumPy vectorized ops |
| `make_Spectrum` + Chronux helpers | 1040–1235 | **No** | Python biomarkers already validated vs MATLAB |
| `plant_band_power.m` | separate file | **No** | Validation helper only |

**Already in Python (reuse):**

- `DbsSpec`, pattern → `pick_dbs_freq` mapping
- `p_beta()` multitaper GPi PSD (13–35 Hz)
- `PlantConfig` (`dt_ms`, `pd`, `corstim`, `neurons_per_region`)
- `IntegrateResult` dataclass

---

## 4. Proposed architecture

```
envs/plant/
  config.py          # unchanged
  dbs.py             # extend: creatdbs waveform generator if not already equivalent
  spikes.py          # spike detection from voltage traces
  biomarkers.py      # unchanged (shared across backends)
  matlab_backend.py  # unchanged
  python_backend.py  # NEW — PythonPlant
  network/           # NEW package
    __init__.py
    integrator.py    # CTX_BG_TH_network main loop
    cells.py         # per-population RHS / channel kinetics
    synapses.py      # connectivity + conductance updates
    gating.py        # *_inf, *_tau helpers (from MATLAB local functions)
```

### `PlantBackend` contract (unchanged)

```python
class PythonPlant:
    def reset(self, seed: int | None = None) -> PythonPlant: ...
    def integrate(self, duration_s: float, dbs_spec: DbsSpec | None = None, *, record_spikes: bool = True) -> IntegrateResult: ...
    def close(self) -> None: ...  # no-op
```

### Configuration surface

- `PlantConfig` remains the single config object.
- New optional field (later): `backend: Literal["matlab", "python"]` in `.rl-dbs.yaml` / `user_config.py` — default **`matlab`** until equivalence gate passes, then flip default to **`python`**.

### Numerics

- **Primary:** NumPy `float64` for the first port (matches MATLAB double).
- **Integration:** Euler or exponential Euler matching MATLAB loop structure in `CTX_BG_TH_network` (verify against reference — do not assume RK without diffing).
- **RNG:** `numpy.random.Generator` seeded from `reset(seed=)` for IC draws (`v = -62 + randn(n)*5` etc.).
- **Optional Phase B:** JAX `jit` + `vmap` over neurons if profiling shows Python loop overhead dominates.

---

## 5. Validation approach

Reuse and extend tests in `tests/envs/`:

| Test | Current | After port |
|------|---------|------------|
| `matlab_plant_test.py` | MATLAB only | Parametrize `backend` |
| `matlab_biomarkers_test.py` | Cross-check Python $P_\beta$ vs MATLAB | Add `PythonPlant` path |
| `matlab_mehregan_env_test.py` | End-to-end env | Same rollouts, both backends |

### Equivalence criteria ([plant.md](../plant.md) §8)

1. **Spikes:** For fixed `seed`, `pd=1`, `pick_dbs_freq=1` (no DBS), GPi spike times per neuron match within **0 ms** (same dt grid) or document if detection threshold differs.
2. **$P_\beta$:** Relative error **< 1%** vs MATLAB-backed run for same spike trains; absolute tolerance **< 0.01** on normalized scale used by Mehregan env.
3. **DBS on:** 130 Hz and 45 Hz pattern indices — $P_\beta$ ordering preserved (CD-DBS lowers beta vs none).
4. **Segment duration:** 2 s (200,000 steps) completes without drift; total simulated time matches.

### CI policy

- Default CI: `pytest -m "not matlab"` uses **mock plant** or fast fixture — unchanged.
- Equivalence job (manual / nightly): `pytest -m matlab` with both backends; requires MATLAB license until Python is trusted.

### Benchmark gate (exit criteria for flipping default)

- [x] `PythonPlant` passes equivalence tests in `tests/envs/` (GPi spikes, $P_\beta$, DBS ordering; seeds 3, 7, 11, 42 with exported init fixtures)
- [ ] One full `mehregan_eval` baseline run (`none`, `cdbs-130hz`) completes with Python backend
- [ ] Documented speedup ≥ **10×** on 2 s integrate (single worker, same hardware as TASK-13 timing)
- [x] [plant.md](../plant.md) §7–§9 updated; [roadmap.md](roadmap.md) Phase 9 status advanced

**2026-07-03 parity resolution:** RNG drift from MATLAB `randperm(n,k)` vs Python `randperm(n)[:k]` is fixed — `PythonPlant.reset(seed)` loads `plant_init_export` fixtures (`tests/fixtures/plant_init_seed{N}.npz`). Fixed-IC and cross-backend tests pass for 2 s segments. `find_spike_times` upward-crossing fix (repolarization no longer counted as spikes) was applied earlier the same day.

**2026-07-03 performance (WSL, seed=42, 2 s, no DBS):** **PythonPlant ≈ 190 s** vs MATLAB **≈ 58–63 s** — **~3× slower** than MATLAB (improved from ~219 s via scalar voltage state, online GPi spike capture, and convolver fast path). **10× speedup gate open** — Numba / inner-loop JIT follow-up.

---

## 6. Implementation phases

### Phase A — Scaffold + DBS waveform (1–2 days)

- Add `python_backend.py` with stub `integrate()` raising `NotImplementedError`.
- Implement `creatdbs` equivalent; unit test against MATLAB-exported `Idbs` vectors (small fixture `.npz`).
- Wire `build_mehregan_env(plant_backend="python")` behind explicit flag (no default flip).

### Phase B — Network integrator (1–2 weeks)

- Port `CTX_BG_TH_network` loop structure: state vectors, synaptic currents, channel updates.
- Port gating functions (`gpe_*`, `stn_*`, `th_*`) as vectorized NumPy.
- Spike detection → `IntegrateResult.gpi_spikes`.
- First equivalence: fixed seed, no DBS, 2 s segment.

### Phase C — Equivalence hardening (3–5 days)

- Parametrize existing matlab tests.
- Tolerance tuning; document any intentional deviations.
- Profile; record baseline timings in [phase4-results.md](phase4-results.md).

### Phase D — Integration + docs (2–3 days)

- `user_config` backend switch; CLI `rl-dbs info plant` shows active backend.
- Update [plant.md](../plant.md) §7 table, [setup.md](../setup.md) (MATLAB optional), [roadmap.md](roadmap.md).
- Optional: `uv run rl-dbs benchmark --plant-backend python`.

**Total estimate:** **3–4 weeks** engineering time (one developer), assuming familiarity with the MATLAB reference. Add **1 week** buffer for numerical debugging.

---

## 7. Risk register

| Risk | Mitigation |
|------|------------|
| Hidden MATLAB state (persistent vars, order-of-ops) | Diff intermediate voltages at single timestep; small-segment tests before 2 s |
| 200k-step loop too slow in pure Python | Profile; consider Numba `@njit` on inner loop or JAX |
| Spike time detection threshold mismatch | Export MATLAB `find_spike_times` output as golden files |
| Memory (7 populations × 10 neurons × 200k steps) | Store spikes only, not full voltage traces, matching current `MatlabPlant` |
| Scope creep (Nguyen 7–35 Hz, adapters) | Mehregan 13–35 Hz gate only; adapters unchanged |

---

## 8. Parallel MATLAB (companion work)

Immediate throughput win — separate child issue for programmer:

- `ProcessPoolExecutor` with **one `matlab.engine` per worker process** (engines are not thread-safe).
- Extend `BenchmarkOptions.workers`, `train_controller(parallel=N)`, `eval_controller(parallel=N)`.
- CLI: `--parallel N` on `train`, `eval`, `benchmark` (documented in [cli.md](../cli.md) but not yet wired).
- Default `N=1`; cap at `min(cpu_count, floor(available_ram_gb / 1.0))` with sensible default **3** on 17 GB host.
- `MatlabPlant(engine=...)` already supports injected engines — worker init creates engine, runs seed(s), exits.

---

## 9. Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-01 | Accelerate native port from Phase 9 | MATLAB step time blocks practical training |
| 2026-07-01 | NumPy first, JAX optional | Matches reference numerics; lowest port risk |
| 2026-07-01 | Keep biomarkers in Python | Already validated; avoids Chronux port |
| 2026-07-01 | Parallel MATLAB first | Days vs weeks; immediate seed parallelism |

---

## 10. References

- Kumaravelu et al. (2016) — bundled MATLAB under `reference-material/KumaraveluEtAl2016/`
- [plant.md](../plant.md) — plant spec §7–9
- [environment.md](../environment.md) — Mehregan Gym API
- [development/testing.md](testing.md) — `@pytest.mark.matlab` policy
