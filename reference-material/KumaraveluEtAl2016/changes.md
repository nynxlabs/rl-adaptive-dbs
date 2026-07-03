# Vendor changes (Kumaravelu et al., 2016 bundle)

Upstream: [ModelDB 206232](https://github.com/ModelDBRepository/206232). Citation and provenance: [`readme.txt`](readme.txt).

**rl-adaptive-dbs** keeps upstream dynamics; these edits support **base MATLAB** and the Phase 2 plant bridge. Do not merge back to ModelDB without review.

---

## 1. `randsample` → `randperm` (`CTX_BG_TH_network`)

**Where:** wiring shuffle for cortical connectivity (`all`, `bll`, … `oll`).

**Upstream:** `randsample(n,n)` — requires **Statistics and Machine Learning Toolbox**.

**Here:** `randperm(n)` — base MATLAB. For fixed `n`, a uniform random permutation of `1:n` (same role as sampling all indices without replacement).

---

## 2. Optional `dynamics_only` (`simulate_network_model`)

**Where:** fifth argument to `simulate_network_model(IT, pd, corstim, pick_dbs_freq, dynamics_only, seed, tmax_ms)`.

**Behavior:** when `dynamics_only` is true, return immediately after `CTX_BG_TH_network`, skipping:

- `make_Spectrum` / inlined `mtspectrumpt` (needs **Signal Processing Toolbox** for `dpss`)
- `.mat` save and `quit`

**Optional args (plant bridge):**

- **`seed`** — when provided, `rng(seed)` for reproducible ICs; otherwise `rng('shuffle')` (upstream default).
- **`tmax_ms`** — segment duration in ms; default **2000**.

**Return values** when `dynamics_only` is true (for `matlab.engine`):

| `nargout` | Value |
|-----------|--------|
| 1 | `gpi_spike_times` — 1×10 cell of GPi spike time vectors (seconds) |
| 2–6 | `dt_ms`, `tmax_ms`, `pd`, `pick_dbs_freq`, `dbs_freq_hz` |
| 7–10 | Optional voltage traces (`vgi`, `vsn`, `vge`, `vstr_indr`) when `nargout` > 6 — for Python parity debugging (`scripts/compare_vgi_trace.py`) |

`spikes_to_cell` packs `find_spike_times` output because nested struct arrays cannot cross the Python engine.

**Why:** Phase 2 calls network dynamics only; Mehregan $P_\beta$ (13–35 Hz) is computed in Python ([`docs/plant.md`](../../docs/plant.md)). `bash scripts/matlab/verify.sh` uses `simulate_network_model(1,1,0,1,true)` (no return values).

**Default:** `dynamics_only = false` — preserves upstream script shape when the fifth argument is omitted.

---

## 3. Optional voltage traces (`dynamics_only`, `nargout` > 6)

**Where:** `simulate_network_model` / `CTX_BG_TH_network` — seventh through tenth outputs when requested.

**Behavior:** Returns full `vgi`, `vsn`, `vge`, `vstr_indr` matrices (neurons × time steps) for parity debugging. Default plant bridge calls use `nargout=6` (unchanged).

**Why:** Localize Python vs MATLAB integrator drift (`scripts/compare_vgi_trace.py`, `scripts/compare_gpe_step5185.py`). Neuron-0 GPe ``Igege`` divergence at step **5185** (~51.85 ms) traces to peer ``S3c`` / GPe spike history (not STN convolver timing); intrinsic GPe drift begins ~3.6 ms on other neurons (2026-07-03).

**`nargout` 11:** optional `gpe_debug_snapshot` struct at MATLAB `i == step + 1` (pre–GPe-update synaptic state for parity scripts).

**`nargout` 12:** `plant_init_export` — voltages, wiring permutations, and heterogeneous conductances captured after the init block (for `scripts/export_plant_init_draws.py`). Must use this path for fixtures; replaying RNG by hand diverges on `randperm(n,k)` vs `randperm(n)[:k]`.

---

## 4. Not changed

- Network equations, integration (`dt = 0.01` ms), DBS waveform construction, inlined multitaper code path (still runs when `dynamics_only` is false).
- Full upstream run still needs Statistics + Signal Processing toolboxes unless further patches are added.

---

## 5. `plant_band_power.m` (Python $P_\beta$ validation)

**Where:** [`plant_band_power.m`](plant_band_power.m) — exported `plant_band_power` / `plant_p_beta` plus inlined Chronux `mtspectrumpt` helpers (copied from `simulate_network_model.m`).

**Why:** Local functions in `simulate_network_model.m` are not callable from `matlab.engine`; this file validates Python `envs.plant.biomarkers.p_beta` in `@pytest.mark.matlab` tests.

**Requires:** **Signal Processing Toolbox** (`dpss`). Skipped when unavailable; Phase 2 biomarkers for training use **Python** by default.
