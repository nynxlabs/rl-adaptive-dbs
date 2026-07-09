# Current replication figures

Working copies of paper figure replications for this effort. Source PNGs live under `figures/` in the vault; regenerate from the repo with the scripts noted below.

---

## Mehregan et al. (paper 1)

### Figure 1b — GPi PSD (healthy vs PD vs 130 Hz cDBS)

![Mehregan Fig 1b replication — mean GPi PSD](figures/fig1b_gpi_psd.png)

| Field | Value |
|-------|-------|
| **Paper** | Mehregan et al. — adaptive DBS / quantization (paper 1) |
| **Panel** | Fig 1b — oscillatory activity / GPi power spectral density |
| **Title** | Oscillatory activity of model neurons in the GPi |
| **Legend** | Healthy Control; PD no Treatment; PD 130 Hz Treatment |
| **Axes** | x: Frequency (Hz), 1–50; y: Power Spectral Density, auto max + 10% headroom |
| **Vault image** | `figures/fig1b_gpi_psd.png` |
| **Repo artifacts** | `artifacts/figures/papers/1/1b/` |
| **Script** | `scripts/figures/papers/1/1b/plot.py` |
| **Spec** | [plant.md](docs/plant.md) §8.1 |
| **Generated** | 2026-07-09 |

**Run:**

```bash
# Simulate + plot (~25 min) — writes curves cache
uv run python scripts/figures/papers/1/1b/plot.py

# Restyle only (~1 s) — no plant simulation
uv run python scripts/figures/papers/1/1b/plot.py --plot-only
uv run python scripts/figures/papers/1/1b/plot.py --plot-only --y-max 100
```

**Defaults:** seeds `0–9` (mean PSD), 10 s segment, Python plant, multitaper PSD (`fpass` 1–100 Hz) averaged over 10 GPi neurons per seed.

**Qualitative gate (seeds 0–9, 2026-07-09):** mean $P_\beta$ orders PD ($\approx 487$) > healthy ($\approx 294$); 130 Hz cDBS suppresses beta ($\approx 164$). Single-seed runs can show sharp healthy beta spikes — seed averaging smooths those.

---

## Mehregan et al. — Fig 5 / Fig 6 (beta time series)

See `scripts/plot_beta_psd_paper_figures.py` and artifacts under `artifacts/ddpg/` (`fig5a_beta_psd_45hz.png`, `fig5b_beta_psd_30hz.png`, etc.). Not yet indexed here — add when promoted to vault.
