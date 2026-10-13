"""Probe: Nguyen Fig 7 step 0 (the reset reading) under the authors' published conventions.

Diagnostic only. Rebuilds each Fig 7 eval episode's reset window (fresh network, the
checkpoint's plant settings, 100 Hz / 300 nA/cm^2 / 0.3 ms for 100 ms) on the 50 eval seeds
and reads GPi alpha-beta two ways:

- ``ours``: ``controllers.snn.reward.alpha_beta_power`` (per-neuron grid over the whole
  100 ms segment, then the mean of per-neuron band powers)
- ``code``: a re-implementation (not an import) of ``make_Spectrum`` / ``mtspectrumpt`` in
  CL-DBS-RL ``src/simulation/simulate_network_optimized.py``: one time grid from the first
  to the last GPi spike of the window, symmetric DPSS (NW 3, K 5), spectra averaged over
  neurons, trapezoid over 7 < f < 35 Hz (strict)

and two reset pulse layouts:

- ``ours``: ``ContinuousPulseTrain`` (30-sample pulses at samples 0, 1000, ..., 10000)
- ``code``: ``creatdbs`` (``int(PW/dt)`` = 29-sample pulses at samples 0, ..., 9000)

Run from the repo root:

    RL_DBS_MAX_THREADS=1 .venv/bin/python scripts/probes/nguyen_fig7_reset_reading.py \
        artifacts/figures/papers/nguyen/4/checkpoint.pt --out /tmp/reset_reading.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402
from scipy.signal.windows import dpss  # noqa: E402

from controllers.snn import adapter as A  # noqa: E402
from controllers.snn.reward import alpha_beta_power  # noqa: E402
from controllers.snn.trainer import load_checkpoint  # noqa: E402
from envs.plant.dbs import DbsSpec  # noqa: E402


def code_alpha_beta(trains: list[np.ndarray], fs: float = 1e5) -> float:
    """CL-DBS-RL make_Spectrum(GPi_APs, Fs=1e5, fpass=[1,100], tapers=[3,5], trialave=1)."""
    nonempty = [np.asarray(t, float) for t in trains if len(t)]
    if not nonempty:
        return 0.0
    mintime = min(float(t.min()) for t in nonempty)
    maxtime = max(float(t.max()) for t in nonempty)
    dt = 1.0 / fs
    t = np.arange(mintime - dt, maxtime + 2 * dt, dt)
    n = len(t)
    nfft = int(max(2 ** np.ceil(np.log2(n)), n))
    f_all = np.arange(0, fs, fs / nfft)[:nfft]
    findx = np.where((f_all >= 1) & (f_all <= 100))[0]
    f = f_all[findx]
    tapers = dpss(n, 3, 5) * np.sqrt(fs)  # (K, N), symmetric (scipy default)
    H = np.fft.fft(tapers.T, nfft, axis=0)[findx, :]
    w = 2 * np.pi * f
    S = np.zeros((len(f), len(trains)))
    for ch, tr in enumerate(trains):
        times = np.asarray(tr, float)
        msp = len(times) / n
        if msp == 0:
            continue
        proj = interp1d(t, tapers, kind="linear", fill_value=0, bounds_error=False)(times)
        J = np.exp(-1j * np.outer(w, times - t[0])) @ proj.T - H * msp
        S[:, ch] = np.mean(np.real(np.conj(J) * J), axis=1)
    S = S.mean(axis=1)
    band = (f > 7) & (f < 35)
    return float(np.trapezoid(S[band], f[band]))


def code_reset_idbs(n_samples: int, *, hz: float, pw_ms: float, amp: float, dt: float = 0.01) -> np.ndarray:
    """CL-DBS-RL creatdbs(pattern, tmax, dt, PW, amplitude)."""
    idbs = np.zeros(n_samples)
    pulse_len = int(pw_ms / dt)
    isi = round((1000.0 / hz) / dt)
    i = 0
    while i < n_samples - 1:
        idbs[i : i + pulse_len] = amp
        i += isi
    return idbs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--spread", type=float, default=None, help="override stn_dbs_spread")
    args = ap.parse_args()

    cfg = load_checkpoint(args.checkpoint, map_location="cpu")["config"].with_variant_defaults()
    cfg = replace(cfg, reset_window_frequency_hz=100.0)
    if args.spread is not None:
        cfg = replace(cfg, stn_dbs_spread=args.spread)
    env = A.NguyenEnvAdapter(config=cfg)
    out: dict = {"checkpoint": str(args.checkpoint), "stn_dbs_spread": cfg.stn_dbs_spread}
    rows = {k: [] for k in ("ours_ours", "ours_code", "code_ours", "code_code")}
    span = []
    for ep in range(args.episodes):
        seed = cfg.seed + 10_000 + ep
        for layout in ("ours", "code"):
            env.reset(seed=seed)  # sets up state; re-run the reset window below to read its spikes
            env._plant.reset(seed=seed)
            env._pulses.reset()
            d = env._dbs
            if layout == "ours":
                env._dbs = d.copy()
                env._dbs.frequency_hz = 100.0
                result = env._integrate_current_dbs()
                env._dbs = d
            else:
                n = int(round(cfg.step_duration_s * 1000.0 / 0.01)) + 1
                idbs = code_reset_idbs(n, hz=100.0, pw_ms=d.pulse_width_ms, amp=d.amplitude)
                result = env._plant.integrate(
                    cfg.step_duration_s,
                    DbsSpec(pick_dbs_freq=2, idbs=idbs, mean_hz=100.0),
                    record_spikes=True, record_th_spikes=True, record_cor_spikes=True,
                    carry=True, **env._bg_kwargs(),
                )
            gpi = env._gpi_spike_trains(result)
            rows[f"{layout}_ours"].append(alpha_beta_power(gpi, duration_s=cfg.step_duration_s, dt_ms=result.dt_ms))
            rows[f"{layout}_code"].append(code_alpha_beta(gpi))
            if layout == "ours":
                allt = np.concatenate([np.asarray(t) for t in gpi if len(t)])
                span.append(float(allt.max() - allt.min()))
    env.close()
    for k, v in rows.items():
        a = np.asarray(v)
        out[k] = {"mean": round(float(a.mean()), 1), "ci95": round(float(1.96 * a.std(ddof=1) / np.sqrt(len(a))), 1),
                  "per_episode": np.round(a, 1).tolist()}
        print(f"pulses={k.split('_')[0]:4s} estimator={k.split('_')[1]:4s} step0 mean {a.mean():.1f} ± {out[k]['ci95']}")
    out["gpi_span_ms"] = {"min": round(1e3 * min(span), 2), "mean": round(1e3 * float(np.mean(span)), 2)}
    print("GPi first-to-last spike span (ms):", out["gpi_span_ms"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
