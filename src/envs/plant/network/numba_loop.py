"""Numba-compiled CBGT integration loop (TASK-17 speed gate).

Falls back to the pure-NumPy loop in ``integrator.py`` when Numba is unavailable
or when debug/trace hooks are requested.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])


MAX_SPIKE_SLOTS = 512
N_CONV = 7
CONV_TH = 0
CONV_STN = 1
CONV_GPE = 2
CONV_GPI = 3
CONV_STR_INDR = 4
CONV_STR_DR = 5
CONV_COR = 6

SPIKE_SYN_THRESHOLD = -10.0
GPI_SPIKE_THRESHOLD = -20.0
TH_SPIKE_THRESHOLD = -20.0


@njit(cache=True)
def _logistic_vec(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if xi >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-xi))
        else:
            ex = np.exp(xi)
            out[i] = ex / (1.0 + ex)
    return out


@njit(cache=True)
def _conv_record_crossings(
    conv_idx: int,
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    thresh: float,
) -> None:
    n = v_prev.size
    for j in range(n):
        if v_prev[j] < thresh and v_curr[j] > thresh:
            c = spike_n[conv_idx, j]
            if c < MAX_SPIKE_SLOTS:
                spike_idx[conv_idx, j, c] = 1
                spike_n[conv_idx, j] = c + 1


@njit(cache=True)
def _conv_record_spike(
    conv_idx: int,
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    neuron: int,
) -> None:
    c = spike_n[conv_idx, neuron]
    if c < MAX_SPIKE_SLOTS:
        spike_idx[conv_idx, neuron, c] = 1
        spike_n[conv_idx, neuron] = c + 1


@njit(cache=True)
def _conv_step_one(
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    max_index: int,
    conv_idx: int,
    n_neurons: int,
) -> None:
    for j in range(n_neurons):
        cnt = spike_n[conv_idx, j]
        new_cnt = 0
        for k in range(cnt):
            idx = spike_idx[conv_idx, j, k] + 1
            if idx < max_index:
                spike_idx[conv_idx, j, new_cnt] = idx
                new_cnt += 1
        spike_n[conv_idx, j] = new_cnt


@njit(cache=True)
def _conv_eval_all(
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    conv_idx: int,
    syn_func: np.ndarray,
    out: np.ndarray,
) -> None:
    n = out.size
    for j in range(n):
        total = 0.0
        cnt = spike_n[conv_idx, j]
        for k in range(cnt):
            total += syn_func[spike_idx[conv_idx, j, k] - 1]
        out[j] = total


@njit(cache=True)
def _conv_step_outputs(
    conv_idx: int,
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    v_prev: np.ndarray,
    v_curr: np.ndarray,
    syn_funcs: tuple,
    max_index: int,
    n_neurons: int,
) -> tuple:
    _conv_record_crossings(conv_idx, spike_idx, spike_n, v_prev, v_curr, SPIKE_SYN_THRESHOLD)
    outputs = []
    for sf in syn_funcs:
        out = np.empty(n_neurons, dtype=np.float64)
        _conv_eval_all(spike_idx, spike_n, conv_idx, sf, out)
        outputs.append(out)
    _conv_step_one(spike_idx, spike_n, max_index, conv_idx, n_neurons)
    return tuple(outputs)


@njit(cache=True, parallel=True)
def run_cbgt_loop(
    n_steps: int,
    dt: float,
    n: int,
    pd: int,
    corstim: int,
    max_index: int,
    idbs: np.ndarray,
    iappco: np.ndarray,
    iappth: np.ndarray,
    ggith: float,
    t_ms: np.ndarray,
    # wiring
    all_idx: np.ndarray,
    bll: np.ndarray,
    cll: np.ndarray,
    dll: np.ndarray,
    ell: np.ndarray,
    fll: np.ndarray,
    gll: np.ndarray,
    hll: np.ndarray,
    ill: np.ndarray,
    jll: np.ndarray,
    kll: np.ndarray,
    lll: np.ndarray,
    mll: np.ndarray,
    nll: np.ndarray,
    oll: np.ndarray,
    _roll_m1: np.ndarray,
    _roll_p1: np.ndarray,
    _roll_m2: np.ndarray,
    _roll_p2: np.ndarray,
    _roll_m3: np.ndarray,
    _roll_m4: np.ndarray,
    _roll_m5: np.ndarray,
    _roll_m6: np.ndarray,
    _roll_m7: np.ndarray,
    _roll_m8: np.ndarray,
    _roll_m9: np.ndarray,
    # heterogeneous conductances
    gcorsna: np.ndarray,
    gcorsnn: np.ndarray,
    gcordrstr: np.ndarray,
    ggege: np.ndarray,
    gsngen: np.ndarray,
    gsngea: np.ndarray,
    gsngi: np.ndarray,
    # synaptic kernels
    syn_th: np.ndarray,
    syn_stn_gpea: np.ndarray,
    syn_stn_gpen: np.ndarray,
    syn_stn_gpi: np.ndarray,
    syn_gpe_stn: np.ndarray,
    syn_gpe_gpi: np.ndarray,
    syn_gpe_gpe: np.ndarray,
    syn_gpi_th: np.ndarray,
    syn_str_indr: np.ndarray,
    syn_str_dr: np.ndarray,
    syn_cor_d2: np.ndarray,
    syn_cor_stn_a: np.ndarray,
    syn_cor_stn_n: np.ndarray,
    # voltages (updated in place)
    vth: np.ndarray,
    vsn: np.ndarray,
    vge: np.ndarray,
    vgi_curr: np.ndarray,
    vstr_indr: np.ndarray,
    vstr_dr: np.ndarray,
    ve: np.ndarray,
    vi: np.ndarray,
    ue: np.ndarray,
    ui: np.ndarray,
    # channel / synaptic state (updated in place)
    H1: np.ndarray,
    R1: np.ndarray,
    N2: np.ndarray,
    H2: np.ndarray,
    M2: np.ndarray,
    A2: np.ndarray,
    B2: np.ndarray,
    C2: np.ndarray,
    D2: np.ndarray,
    D1: np.ndarray,
    P2: np.ndarray,
    Q2: np.ndarray,
    R2: np.ndarray,
    CAsn2: np.ndarray,
    N3: np.ndarray,
    H3: np.ndarray,
    R3: np.ndarray,
    CA3: np.ndarray,
    N4: np.ndarray,
    H4: np.ndarray,
    R4: np.ndarray,
    CA4: np.ndarray,
    m5: np.ndarray,
    h5: np.ndarray,
    n5: np.ndarray,
    p5: np.ndarray,
    m6: np.ndarray,
    h6: np.ndarray,
    n6: np.ndarray,
    p6: np.ndarray,
    S2a: np.ndarray,
    S21a: np.ndarray,
    S2b: np.ndarray,
    S2an: np.ndarray,
    S21an: np.ndarray,
    S3a: np.ndarray,
    S31a: np.ndarray,
    S3b: np.ndarray,
    S31b: np.ndarray,
    S32b: np.ndarray,
    S3c: np.ndarray,
    S31c: np.ndarray,
    S32c: np.ndarray,
    S4: np.ndarray,
    S5: np.ndarray,
    S51: np.ndarray,
    S52: np.ndarray,
    S53: np.ndarray,
    S54: np.ndarray,
    S55: np.ndarray,
    S56: np.ndarray,
    S57: np.ndarray,
    S58: np.ndarray,
    S59: np.ndarray,
    S9: np.ndarray,
    S6a: np.ndarray,
    S6b: np.ndarray,
    S6bn: np.ndarray,
    S61b: np.ndarray,
    S61bn: np.ndarray,
    S91: np.ndarray,
    S92: np.ndarray,
    S93: np.ndarray,
    S94: np.ndarray,
    S95: np.ndarray,
    S96: np.ndarray,
    S97: np.ndarray,
    S98: np.ndarray,
    S99: np.ndarray,
    S7: np.ndarray,
    S8: np.ndarray,
    S1a: np.ndarray,
    S1b: np.ndarray,
    S1c: np.ndarray,
    Z1a: np.ndarray,
    Z1b: np.ndarray,
    # convolver state
    spike_idx: np.ndarray,
    spike_n: np.ndarray,
    # GPi spike output
    gpi_spike_buf: np.ndarray,
    gpi_spike_n: np.ndarray,
    th_spike_buf: np.ndarray,
    th_spike_n: np.ndarray,
    record_th_spikes: bool,
    cor_spike_buf: np.ndarray,
    cor_spike_n: np.ndarray,
    record_cor_spikes: bool,
    # scalars
    iappgpe: float,
    uce_scale: float,
) -> None:
    v1_prev = np.empty(n, dtype=np.float64)
    v2_prev = np.empty(n, dtype=np.float64)
    v3_prev = np.empty(n, dtype=np.float64)
    v4_prev = np.empty(n, dtype=np.float64)
    v5_prev = np.empty(n, dtype=np.float64)
    v6_prev = np.empty(n, dtype=np.float64)
    v7_prev = np.empty(n, dtype=np.float64)
    v8_prev = np.empty(n, dtype=np.float64)
    uce = np.zeros(n, dtype=np.float64)
    uci = np.zeros(n, dtype=np.float64)
    S21a_w = np.empty(n, dtype=np.float64)
    S21an_w = np.empty(n, dtype=np.float64)
    S21b_w = np.empty(n, dtype=np.float64)
    S31a_w = np.empty(n, dtype=np.float64)
    S31b_w = np.empty(n, dtype=np.float64)
    S31c_w = np.empty(n, dtype=np.float64)
    S32c_w = np.empty(n, dtype=np.float64)
    S32b_w = np.empty(n, dtype=np.float64)
    S11cr_w = np.empty(n, dtype=np.float64)
    S12cr_w = np.empty(n, dtype=np.float64)
    S13cr_w = np.empty(n, dtype=np.float64)
    S14cr_w = np.empty(n, dtype=np.float64)
    S11br_w = np.empty(n, dtype=np.float64)
    S12br_w = np.empty(n, dtype=np.float64)
    S13br_w = np.empty(n, dtype=np.float64)
    S14br_w = np.empty(n, dtype=np.float64)
    S11ar_w = np.empty(n, dtype=np.float64)
    S12ar_w = np.empty(n, dtype=np.float64)
    S13ar_w = np.empty(n, dtype=np.float64)
    S14ar_w = np.empty(n, dtype=np.float64)
    S81r_w = np.empty(n, dtype=np.float64)
    S82r_w = np.empty(n, dtype=np.float64)
    S83r_w = np.empty(n, dtype=np.float64)
    S51_w = np.empty(n, dtype=np.float64)
    S52_w = np.empty(n, dtype=np.float64)
    S53_w = np.empty(n, dtype=np.float64)
    S54_w = np.empty(n, dtype=np.float64)
    S55_w = np.empty(n, dtype=np.float64)
    S56_w = np.empty(n, dtype=np.float64)
    S57_w = np.empty(n, dtype=np.float64)
    S58_w = np.empty(n, dtype=np.float64)
    S59_w = np.empty(n, dtype=np.float64)
    S61b_w = np.empty(n, dtype=np.float64)
    S61bn_w = np.empty(n, dtype=np.float64)
    S91_w = np.empty(n, dtype=np.float64)
    S92_w = np.empty(n, dtype=np.float64)
    S93_w = np.empty(n, dtype=np.float64)
    S94_w = np.empty(n, dtype=np.float64)
    S95_w = np.empty(n, dtype=np.float64)
    S96_w = np.empty(n, dtype=np.float64)
    S97_w = np.empty(n, dtype=np.float64)
    S98_w = np.empty(n, dtype=np.float64)
    S99_w = np.empty(n, dtype=np.float64)
    S7_old = np.empty(n, dtype=np.float64)
    S2a_old = np.empty(n, dtype=np.float64)
    S2an_old = np.empty(n, dtype=np.float64)
    S2b_old = np.empty(n, dtype=np.float64)

    _CM = 1.0
    _AE = 0.02
    _BE = 0.2
    _CE = -65.0
    _DE = 8.0
    _AI = 0.1
    _BI = 0.2
    _CI = -65.0
    _DI = 2.0
    _GL0 = 0.05
    _GL1 = 0.35
    _GL2 = 0.1
    _GL3 = 0.1
    _EL0 = -70.0
    _EL1 = -60.0
    _EL2 = -65.0
    _EL3 = -67.0
    _GNA0 = 3.0
    _GNA1 = 49.0
    _GNA2 = 120.0
    _GNA3 = 100.0
    _ENA0 = 50.0
    _ENA1 = 60.0
    _ENA2 = 55.0
    _ENA3 = 50.0
    _GK0 = 5.0
    _GK1 = 57.0
    _GK2 = 30.0
    _GK3 = 80.0
    _EK0 = -75.0
    _EK1 = -90.0
    _EK2 = -80.0
    _EK3 = -100.0
    _GT0 = 5.0
    _GT1 = 5.0
    _GT2 = 0.5
    _ET = 0.0
    _GCA2 = 0.15
    _ECA2 = 120.0
    _EM = -100.0
    _GAHP2 = 10.0
    _K1_GPE = 10.0
    _KCA2 = 15.0
    _GA = 5.0
    _GL_STN = 15.0
    _GCAK = 1.0
    _KCA_STN = 2e-3
    _ALP = 1.0 / (2.0 * 96485.0)
    _CON = (8314.0 * 298.0) / (2.0 * 96485.0)
    _CAO = 2000.0
    _CA_MIN = 1e-8
    _ESYN0 = -85.0
    _ESYN1 = 0.0
    _ESYN2 = -85.0
    _ESYN3 = 0.0
    _ESYN4 = -85.0
    _ESYN5 = -85.0
    _ESYN6 = -80.0
    _TAU = 5.0
    _TAU_I = 13.0
    _GGITH = ggith
    _GGESN = 0.5
    _GSTRGPE = 0.5
    _GSTRGPI = 0.5
    _GGIGI = 0.5
    _GM = 1.0
    _GGABA = 0.1
    _GCORINDRSTR = 0.07
    _GIE = 0.2
    _GTHCOR = 0.15
    _GEi = 0.1
    _IAPPGPI = 3.0
    _STN_TD2 = 130.0
    _STN_TR2 = 2.0
    _GPE_TR = 30.0

    for step in range(1, n_steps):
        for i in range(n):
            v1_prev[i] = vth[i]
            v2_prev[i] = vsn[i]
            v3_prev[i] = vge[i]
            v4_prev[i] = vgi_curr[i]
            v5_prev[i] = vstr_indr[i]
            v6_prev[i] = vstr_dr[i]
            v7_prev[i] = ve[i]
            v8_prev[i] = vi[i]

        # wiring shifts
        for i in range(n):
            S21a_w[i] = S2a[_roll_p1[i]]
            S21an_w[i] = S2an[_roll_p1[i]]
            S21b_w[i] = S2b[_roll_p1[i]]
            S31a_w[i] = S3a[_roll_m1[i]]
            S31b_w[i] = S3b[_roll_m1[i]]
            S31c_w[i] = S3c[_roll_m1[i]]
            S32c_w[i] = S3c[_roll_p2[i]]
            S32b_w[i] = S3b[_roll_p2[i]]
            S51_w[i] = S5[_roll_m1[i]]
            S52_w[i] = S5[_roll_m2[i]]
            S53_w[i] = S5[_roll_m3[i]]
            S54_w[i] = S5[_roll_m4[i]]
            S55_w[i] = S5[_roll_m5[i]]
            S56_w[i] = S5[_roll_m6[i]]
            S57_w[i] = S5[_roll_m7[i]]
            S58_w[i] = S5[_roll_m8[i]]
            S59_w[i] = S5[_roll_m9[i]]
            S61b_w[i] = S6b[_roll_m1[i]]
            S61bn_w[i] = S6bn[_roll_m1[i]]
            S91_w[i] = S9[_roll_m1[i]]
            S92_w[i] = S9[_roll_m2[i]]
            S93_w[i] = S9[_roll_m3[i]]
            S94_w[i] = S9[_roll_m4[i]]
            S95_w[i] = S9[_roll_m5[i]]
            S96_w[i] = S9[_roll_m6[i]]
            S97_w[i] = S9[_roll_m7[i]]
            S98_w[i] = S9[_roll_m8[i]]
            S99_w[i] = S9[_roll_m9[i]]

        for i in range(n):
            S11cr_w[i] = S1c[all_idx[i]]
            S12cr_w[i] = S1c[bll[i]]
            S13cr_w[i] = S1c[cll[i]]
            S14cr_w[i] = S1c[dll[i]]
            S11br_w[i] = S1b[ell[i]]
            S12br_w[i] = S1b[fll[i]]
            S13br_w[i] = S1b[gll[i]]
            S14br_w[i] = S1b[hll[i]]
            S11ar_w[i] = S1a[ill[i]]
            S12ar_w[i] = S1a[jll[i]]
            S13ar_w[i] = S1a[kll[i]]
            S14ar_w[i] = S1a[lll[i]]
            S81r_w[i] = S8[mll[i]]
            S82r_w[i] = S8[nll[i]]
            S83r_w[i] = S8[oll[i]]

        for i in range(n):
            S7_old[i] = S7[i]
            S2a_old[i] = S2a[i]
            S2an_old[i] = S2an[i]
            S2b_old[i] = S2b[i]

        # --- Thalamus (all neurons, then conv) ---
        for i in range(n):
            V1 = v1_prev[i]
            m1_i = 1.0 / (1.0 + np.exp(-(V1 + 37.0) / 7.0))
            h1_i = 1.0 / (1.0 + np.exp((V1 + 41.0) / 4.0))
            p1_i = 1.0 / (1.0 + np.exp(-(V1 + 60.0) / 6.2))
            r1_i = 1.0 / (1.0 + np.exp((V1 + 84.0) / 4.0))
            ah_i = 0.128 * np.exp(-(V1 + 46.0) / 18.0)
            bh_i = 4.0 / (1.0 + np.exp(-(V1 + 23.0) / 5.0))
            th1_i = 1.0 / (ah_i + bh_i)
            tr1_i = 0.15 * (28.0 + np.exp(-(V1 + 25.0) / 10.5))
            il1 = _GL0 * (V1 - _EL0)
            ina1 = _GNA0 * (m1_i**3) * H1[i] * (V1 - _ENA0)
            ik1 = _GK0 * ((0.75 * (1.0 - H1[i])) ** 4) * (V1 - _EK0)
            it1 = _GT0 * (p1_i**2) * R1[i] * (V1 - _ET)
            igith = _GGITH * (V1 - _ESYN5) * S4[i]
            vth[i] = V1 + dt * ((1.0 / _CM) * (-il1 - ik1 - ina1 - it1 - igith + iappth[step]))
            if record_th_spikes:
                if V1 <= TH_SPIKE_THRESHOLD and vth[i] > TH_SPIKE_THRESHOLD:
                    cnt_th = th_spike_n[i]
                    if cnt_th < th_spike_buf.shape[1]:
                        th_spike_buf[i, cnt_th] = t_ms[step - 1] / 1000.0
                        th_spike_n[i] = cnt_th + 1
            H1[i] = H1[i] + dt * ((h1_i - H1[i]) / th1_i)
            R1[i] = R1[i] + dt * ((r1_i - R1[i]) / tr1_i)
        _conv_record_crossings(CONV_TH, spike_idx, spike_n, v1_prev, vth, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_TH, syn_th, S7)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_TH, n)

        # --- STN ---
        for i in range(n):
            V2 = v2_prev[i]
            n2_i = 1.0 / (1.0 + np.exp(-(V2 + 41.0) / 14.0))
            m2_i = 1.0 / (1.0 + np.exp(-(V2 + 40.0) / 8.0))
            h2_i = 1.0 / (1.0 + np.exp((V2 + 45.5) / 6.4))
            a2_i = 1.0 / (1.0 + np.exp(-(V2 + 45.0) / 14.7))
            b2_i = 1.0 / (1.0 + np.exp((V2 + 90.0) / 7.5))
            c2_i = 1.0 / (1.0 + np.exp(-(V2 + 30.6) / 5.0))
            x_d2 = -(V2 - 0.1) / 0.02
            if x_d2 >= 0.0:
                d2_i = 1.0 / (1.0 + np.exp(-x_d2))
            else:
                d2_i = np.exp(x_d2) / (1.0 + np.exp(x_d2))
            d1_i = 1.0 / (1.0 + np.exp((V2 + 60.0) / 7.5))
            p2_i = 1.0 / (1.0 + np.exp(-(V2 + 56.0) / 6.7))
            q2_i = 1.0 / (1.0 + np.exp((V2 + 85.0) / 5.8))
            x_r2 = (V2 - 0.17) / 0.08
            if x_r2 >= 0.0:
                r2_i = 1.0 / (1.0 + np.exp(-x_r2))
            else:
                r2_i = np.exp(x_r2) / (1.0 + np.exp(x_r2))
            tn2_i = 11.0 / (np.exp(-(V2 + 40.0) / -40.0) + np.exp(-(V2 + 40.0) / 50.0))
            tm2_i = 0.2 + 3.0 / (1.0 + np.exp(-(V2 + 53.0) / -0.7))
            th2_i = 24.5 / (np.exp(-(V2 + 50.0) / -15.0) + np.exp(-(V2 + 50.0) / 16.0))
            ta2_i = 1.0 + 1.0 / (1.0 + np.exp(-(V2 + 40.0) / -0.5))
            tb2_i = 200.0 / (np.exp(-(V2 + 60.0) / -30.0) + np.exp(-(V2 + 40.0) / 10.0))
            tc2_i = 45.0 + 10.0 / (np.exp(-(V2 + 27.0) / -20.0) + np.exp(-(V2 + 50.0) / 15.0))
            td1_i = 400.0 + 500.0 / (np.exp(-(V2 + 40.0) / -15.0) + np.exp(-(V2 + 20.0) / 20.0))
            tp2_i = 5.0 + 0.33 / (np.exp(-(V2 + 27.0) / -10.0) + np.exp(-(V2 + 102.0) / 15.0))
            tq2_i = 400.0 / (np.exp(-(V2 + 50.0) / -15.0) + np.exp(-(V2 + 50.0) / 16.0))
            casn_i = CAsn2[i]
            if casn_i < _CA_MIN:
                casn_i = _CA_MIN
            ecasn_i = _CON * np.log(_CAO / casn_i)
            ina2 = _GNA1 * (M2[i] ** 3) * H2[i] * (V2 - _ENA1)
            ik2 = _GK1 * (N2[i] ** 4) * (V2 - _EK1)
            ia2 = _GA * (A2[i] ** 2) * B2[i] * (V2 - _EK1)
            il2_stn = _GL_STN * (C2[i] ** 2) * D1[i] * D2[i] * (V2 - ecasn_i)
            it2 = _GT1 * (P2[i] ** 2) * Q2[i] * (V2 - ecasn_i)
            icak2 = _GCAK * (R2[i] ** 2) * (V2 - _EK1)
            il2 = _GL1 * (V2 - _EL1)
            igesn = _GGESN * ((V2 - _ESYN0) * (S3a[i] + S31a_w[i]))
            icorsnampa = gcorsna[i] * (V2 - _ESYN1) * (S6b[i] + S61b_w[i])
            icorsnnmda = gcorsnn[i] * (V2 - _ESYN1) * (S6bn[i] + S61bn_w[i])
            vsn[i] = V2 + dt * ((1.0 / _CM) * (-ina2 - ik2 - ia2 - il2_stn - it2 - icak2 - il2 - igesn - icorsnampa - icorsnnmda + idbs[step]))
            N2[i] = N2[i] + dt * ((n2_i - N2[i]) / tn2_i)
            H2[i] = H2[i] + dt * ((h2_i - H2[i]) / th2_i)
            M2[i] = M2[i] + dt * ((m2_i - M2[i]) / tm2_i)
            A2[i] = A2[i] + dt * ((a2_i - A2[i]) / ta2_i)
            B2[i] = B2[i] + dt * ((b2_i - B2[i]) / tb2_i)
            C2[i] = C2[i] + dt * ((c2_i - C2[i]) / tc2_i)
            D2[i] = D2[i] + dt * ((d2_i - D2[i]) / _STN_TD2)
            D1[i] = D1[i] + dt * ((d1_i - D1[i]) / td1_i)
            P2[i] = P2[i] + dt * ((p2_i - P2[i]) / tp2_i)
            Q2[i] = Q2[i] + dt * ((q2_i - Q2[i]) / tq2_i)
            R2[i] = R2[i] + dt * ((r2_i - R2[i]) / _STN_TR2)
            CAsn2[i] = CAsn2[i] + dt * ((-_ALP * (il2_stn + it2)) - (_KCA_STN * CAsn2[i]))
            if CAsn2[i] < _CA_MIN:
                CAsn2[i] = _CA_MIN
        _conv_record_crossings(CONV_STN, spike_idx, spike_n, v2_prev, vsn, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_STN, syn_stn_gpea, S2a)
        _conv_eval_all(spike_idx, spike_n, CONV_STN, syn_stn_gpen, S2an)
        _conv_eval_all(spike_idx, spike_n, CONV_STN, syn_stn_gpi, S2b)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_STN, n)

        # --- GPe ---
        for i in range(n):
            V3 = v3_prev[i]
            m3_i = 1.0 / (1.0 + np.exp(-(V3 + 37.0) / 10.0))
            n3_i = 1.0 / (1.0 + np.exp(-(V3 + 50.0) / 14.0))
            h3_i = 1.0 / (1.0 + np.exp((V3 + 58.0) / 12.0))
            a3_i = 1.0 / (1.0 + np.exp(-(V3 + 57.0) / 2.0))
            s3_i = 1.0 / (1.0 + np.exp(-(V3 + 35.0) / 2.0))
            r3_i = 1.0 / (1.0 + np.exp((V3 + 70.0) / 2.0))
            tn3_i = 0.05 + 0.27 / (1.0 + np.exp(-(V3 + 40.0) / -12.0))
            th3_i = 0.05 + 0.27 / (1.0 + np.exp(-(V3 + 40.0) / -12.0))
            il3 = _GL2 * (V3 - _EL2)
            ik3 = _GK2 * (N3[i] ** 4) * (V3 - _EK2)
            ina3 = _GNA2 * (m3_i**3) * H3[i] * (V3 - _ENA2)
            it3 = _GT2 * (a3_i**3) * R3[i] * (V3 - _ECA2)
            ica3 = _GCA2 * (s3_i**2) * (V3 - _ECA2)
            ca3_i = CA3[i]
            if ca3_i < _CA_MIN:
                ca3_i = _CA_MIN
            iahp3 = _GAHP2 * (V3 - _EK2) * (ca3_i / (ca3_i + _K1_GPE))
            isngeampa = gsngea[i] * ((V3 - _ESYN1) * (S2a_old[i] + S21a_w[i]))
            isngenmda = gsngen[i] * ((V3 - _ESYN1) * (S2an_old[i] + S21an_w[i]))
            igege = (0.25 * (pd * 3 + 1)) * ggege[i] * ((V3 - _ESYN2) * (S31c_w[i] + S32c_w[i]))
            istrgpe = _GSTRGPE * (V3 - _ESYN5) * (S5[i] + S51_w[i] + S52_w[i] + S53_w[i] + S54_w[i] + S55_w[i] + S56_w[i] + S57_w[i] + S58_w[i] + S59_w[i])
            vge[i] = V3 + dt * ((1.0 / _CM) * (-il3 - ik3 - ina3 - it3 - ica3 - iahp3 - isngeampa - isngenmda - igege - istrgpe + iappgpe))
            N3[i] = N3[i] + dt * (0.1 * (n3_i - N3[i]) / tn3_i)
            H3[i] = H3[i] + dt * (0.05 * (h3_i - H3[i]) / th3_i)
            R3[i] = R3[i] + dt * (1.0 * (r3_i - R3[i]) / _GPE_TR)
            CA3[i] = CA3[i] + dt * (1e-4 * (-ica3 - it3 - _KCA2 * CA3[i]))
            if CA3[i] < _CA_MIN:
                CA3[i] = _CA_MIN
        _conv_record_crossings(CONV_GPE, spike_idx, spike_n, v3_prev, vge, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_GPE, syn_gpe_stn, S3a)
        _conv_eval_all(spike_idx, spike_n, CONV_GPE, syn_gpe_gpi, S3b)
        _conv_eval_all(spike_idx, spike_n, CONV_GPE, syn_gpe_gpe, S3c)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_GPE, n)

        # --- GPi ---
        for i in range(n):
            V4 = v4_prev[i]
            m4_i = 1.0 / (1.0 + np.exp(-(V4 + 37.0) / 10.0))
            n4_i = 1.0 / (1.0 + np.exp(-(V4 + 50.0) / 14.0))
            h4_i = 1.0 / (1.0 + np.exp((V4 + 58.0) / 12.0))
            a4_i = 1.0 / (1.0 + np.exp(-(V4 + 57.0) / 2.0))
            s4_i = 1.0 / (1.0 + np.exp(-(V4 + 35.0) / 2.0))
            r4_i = 1.0 / (1.0 + np.exp((V4 + 70.0) / 2.0))
            tn4_i = 0.05 + 0.27 / (1.0 + np.exp(-(V4 + 40.0) / -12.0))
            th4_i = 0.05 + 0.27 / (1.0 + np.exp(-(V4 + 40.0) / -12.0))
            il4 = _GL2 * (V4 - _EL2)
            ik4 = _GK2 * (N4[i] ** 4) * (V4 - _EK2)
            ina4 = _GNA2 * (m4_i**3) * H4[i] * (V4 - _ENA2)
            it4 = _GT2 * (a4_i**3) * R4[i] * (V4 - _ECA2)
            ica4 = _GCA2 * (s4_i**2) * (V4 - _ECA2)
            ca4_i = CA4[i]
            if ca4_i < _CA_MIN:
                ca4_i = _CA_MIN
            iahp4 = _GAHP2 * (V4 - _EK2) * (ca4_i / (ca4_i + _K1_GPE))
            isngi = gsngi[i] * ((V4 - _ESYN3) * (S2b_old[i] + S21b_w[i]))
            igigi = _GGIGI * ((V4 - _ESYN4) * (S31b_w[i] + S32b_w[i]))
            istrgpi = _GSTRGPI * (V4 - _ESYN5) * (S9[i] + S91_w[i] + S92_w[i] + S93_w[i] + S94_w[i] + S95_w[i] + S96_w[i] + S97_w[i] + S98_w[i] + S99_w[i])
            vgi_curr[i] = V4 + dt * ((1.0 / _CM) * (-il4 - ik4 - ina4 - it4 - ica4 - iahp4 - isngi - igigi - istrgpi + _IAPPGPI))
            if V4 <= GPI_SPIKE_THRESHOLD and vgi_curr[i] > GPI_SPIKE_THRESHOLD:
                cnt = gpi_spike_n[i]
                if cnt < gpi_spike_buf.shape[1]:
                    gpi_spike_buf[i, cnt] = t_ms[step - 1] / 1000.0
                    gpi_spike_n[i] = cnt + 1
            N4[i] = N4[i] + dt * (0.1 * (n4_i - N4[i]) / tn4_i)
            H4[i] = H4[i] + dt * (0.05 * (h4_i - H4[i]) / th4_i)
            R4[i] = R4[i] + dt * (1.0 * (r4_i - R4[i]) / _GPE_TR)
            CA4[i] = CA4[i] + dt * (1e-4 * (-ica4 - it4 - _KCA2 * CA4[i]))
            if CA4[i] < _CA_MIN:
                CA4[i] = _CA_MIN
        _conv_record_crossings(CONV_GPI, spike_idx, spike_n, v4_prev, vgi_curr, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_GPI, syn_gpi_th, S4)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_GPI, n)

        # --- Striatum D2 ---
        for i in range(n):
            V5 = v5_prev[i]
            ina5 = _GNA3 * (m5[i] ** 3) * h5[i] * (V5 - _ENA3)
            ik5 = _GK3 * (n5[i] ** 4) * (V5 - _EK3)
            il5 = _GL3 * (V5 - _EL3)
            im5 = (2.6 - 1.1 * pd) * _GM * p5[i] * (V5 - _EM)
            igaba5 = (_GGABA / 4.0) * (V5 - _ESYN6) * (S11cr_w[i] + S12cr_w[i] + S13cr_w[i] + S14cr_w[i])
            icorstr5 = _GCORINDRSTR * (V5 - _ESYN1) * S6a[i]
            vstr_indr[i] = V5 + (dt / _CM) * (-ina5 - ik5 - il5 - im5 - igaba5 - icorstr5)
            am5 = (0.32 * (54.0 + V5)) / (1.0 - np.exp((-54.0 - V5) / 4.0))
            bm5 = 0.28 * (V5 + 27.0) / (np.exp((27.0 + V5) / 5.0) - 1.0)
            an5 = (0.032 * (52.0 + V5)) / (1.0 - np.exp((-52.0 - V5) / 5.0))
            bn5 = 0.5 * np.exp((-57.0 - V5) / 40.0)
            ap5 = (3.209e-4 * (30.0 + V5)) / (1.0 - np.exp((-30.0 - V5) / 9.0))
            bp5 = (-3.209e-4 * (30.0 + V5)) / (1.0 - np.exp((30.0 + V5) / 9.0))
            ah5 = 0.128 * np.exp((-50.0 - V5) / 18.0)
            bh5 = 4.0 / (1.0 + np.exp((-27.0 - V5) / 5.0))
            m5[i] = m5[i] + dt * (am5 * (1.0 - m5[i]) - bm5 * m5[i])
            h5[i] = h5[i] + dt * (ah5 * (1.0 - h5[i]) - bh5 * h5[i])
            n5[i] = n5[i] + dt * (an5 * (1.0 - n5[i]) - bn5 * n5[i])
            p5[i] = p5[i] + dt * (ap5 * (1.0 - p5[i]) - bp5 * p5[i])
            gg5 = 2.0 * (1.0 + np.tanh(V5 / 4.0))
            S1c[i] = S1c[i] + dt * ((gg5 * (1.0 - S1c[i])) - (S1c[i] / _TAU_I))
        _conv_record_crossings(CONV_STR_INDR, spike_idx, spike_n, v5_prev, vstr_indr, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_STR_INDR, syn_str_indr, S5)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_STR_INDR, n)

        # --- Striatum D1 ---
        for i in range(n):
            V6 = v6_prev[i]
            ina6 = _GNA3 * (m6[i] ** 3) * h6[i] * (V6 - _ENA3)
            ik6 = _GK3 * (n6[i] ** 4) * (V6 - _EK3)
            il6 = _GL3 * (V6 - _EL3)
            im6 = (2.6 - 1.1 * pd) * _GM * p6[i] * (V6 - _EM)
            igaba6 = (_GGABA / 3.0) * (V6 - _ESYN6) * (S81r_w[i] + S82r_w[i] + S83r_w[i])
            icorstr6 = gcordrstr[i] * (V6 - _ESYN1) * S6a[i]
            vstr_dr[i] = V6 + (dt / _CM) * (-ina6 - ik6 - il6 - im6 - igaba6 - icorstr6)
            am6 = (0.32 * (54.0 + V6)) / (1.0 - np.exp((-54.0 - V6) / 4.0))
            bm6 = 0.28 * (V6 + 27.0) / (np.exp((27.0 + V6) / 5.0) - 1.0)
            an6 = (0.032 * (52.0 + V6)) / (1.0 - np.exp((-52.0 - V6) / 5.0))
            bn6 = 0.5 * np.exp((-57.0 - V6) / 40.0)
            ap6 = (3.209e-4 * (30.0 + V6)) / (1.0 - np.exp((-30.0 - V6) / 9.0))
            bp6 = (-3.209e-4 * (30.0 + V6)) / (1.0 - np.exp((30.0 + V6) / 9.0))
            ah6 = 0.128 * np.exp((-50.0 - V6) / 18.0)
            bh6 = 4.0 / (1.0 + np.exp((-27.0 - V6) / 5.0))
            m6[i] = m6[i] + dt * (am6 * (1.0 - m6[i]) - bm6 * m6[i])
            h6[i] = h6[i] + dt * (ah6 * (1.0 - h6[i]) - bh6 * h6[i])
            n6[i] = n6[i] + dt * (an6 * (1.0 - n6[i]) - bn6 * n6[i])
            p6[i] = p6[i] + dt * (ap6 * (1.0 - p6[i]) - bp6 * p6[i])
            gg6 = 2.0 * (1.0 + np.tanh(V6 / 4.0))
            S8[i] = S8[i] + dt * ((gg6 * (1.0 - S8[i])) - (S8[i] / _TAU_I))
        _conv_record_crossings(CONV_STR_DR, spike_idx, spike_n, v6_prev, vstr_dr, SPIKE_SYN_THRESHOLD)
        _conv_eval_all(spike_idx, spike_n, CONV_STR_DR, syn_str_dr, S9)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_STR_DR, n)

        # --- Cortex exc (currents from prev-step S7) ---
        for i in range(n):
            V7 = v7_prev[i]
            iie = _GIE * (V7 - _ESYN0) * (S11br_w[i] + S12br_w[i] + S13br_w[i] + S14br_w[i])
            ithcor = _GTHCOR * (V7 - _ESYN1) * S7_old[i]
            ve_i = V7 + dt * ((0.04 * (V7**2)) + (5.0 * V7) + 140.0 - ue[i] - iie - ithcor + iappco[step])
            ue_i = ue[i] + dt * (_AE * ((_BE * V7) - ue[i]))
            if V7 >= 30.0:
                ve_i = _CE
                ue_i = ue[i] + _DE
                _conv_record_spike(CONV_COR, spike_idx, spike_n, i)
            ve[i] = ve_i
            ue[i] = ue_i
            if record_cor_spikes:
                if v7_prev[i] <= TH_SPIKE_THRESHOLD and ve_i > TH_SPIKE_THRESHOLD:
                    idx = i
                    cnt = cor_spike_n[idx]
                    if cnt < cor_spike_buf.shape[1]:
                        cor_spike_buf[idx, cnt] = t_ms[step - 1] / 1000.0
                        cor_spike_n[idx] = cnt + 1
        _conv_eval_all(spike_idx, spike_n, CONV_COR, syn_cor_d2, S6a)
        _conv_eval_all(spike_idx, spike_n, CONV_COR, syn_cor_stn_a, S6b)
        _conv_eval_all(spike_idx, spike_n, CONV_COR, syn_cor_stn_n, S6bn)
        _conv_step_one(spike_idx, spike_n, max_index, CONV_COR, n)

        # --- Cortex inh + syn filters ---
        for i in range(n):
            V7 = v7_prev[i]
            V8 = v8_prev[i]
            iei = _GEi * (V8 - _ESYN1) * (S11ar_w[i] + S12ar_w[i] + S13ar_w[i] + S14ar_w[i])
            vi_i = V8 + dt * ((0.04 * (V8**2)) + (5.0 * V8) + 140.0 - ui[i] - iei + iappco[step])
            ui_i = ui[i] + dt * (_AI * ((_BI * V8) - ui[i]))
            if V8 >= 30.0:
                vi_i = _CI
                ui_i = ui[i] + _DI
            vi[i] = vi_i
            ui[i] = ui_i
            if record_cor_spikes:
                if v8_prev[i] <= TH_SPIKE_THRESHOLD and vi_i > TH_SPIKE_THRESHOLD:
                    idx = n + i
                    cnt = cor_spike_n[idx]
                    if cnt < cor_spike_buf.shape[1]:
                        cor_spike_buf[idx, cnt] = t_ms[step - 1] / 1000.0
                        cor_spike_n[idx] = cnt + 1
            if V7 < SPIKE_SYN_THRESHOLD and ve[i] > SPIKE_SYN_THRESHOLD:
                uce[i] = uce_scale
            else:
                uce[i] = 0.0
            S1a[i] = S1a[i] + dt * Z1a[i]
            z1adot = uce[i] - (2.0 / _TAU) * Z1a[i] - (1.0 / (_TAU**2)) * S1a[i]
            Z1a[i] = Z1a[i] + dt * z1adot
            if V8 < SPIKE_SYN_THRESHOLD and vi[i] > SPIKE_SYN_THRESHOLD:
                uci[i] = uce_scale
            else:
                uci[i] = 0.0
            S1b[i] = S1b[i] + dt * Z1b[i]
            z1bdot = uci[i] - (2.0 / _TAU) * Z1b[i] - (1.0 / (_TAU**2)) * S1b[i]
            Z1b[i] = Z1b[i] + dt * z1bdot



def numba_loop_available() -> bool:
    return _NUMBA_AVAILABLE


def gpi_spikes_from_buffer(
    gpi_spike_buf: np.ndarray,
    gpi_spike_n: np.ndarray,
) -> list[np.ndarray]:
    spikes: list[np.ndarray] = []
    for i in range(gpi_spike_buf.shape[0]):
        count = int(gpi_spike_n[i])
        spikes.append(gpi_spike_buf[i, :count].copy())
    return spikes
