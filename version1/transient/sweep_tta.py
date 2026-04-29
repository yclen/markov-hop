import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from data_io import save_sweep
from transient_pulse import run_tta


# ── Parameters ────────────────────────────────────────────────────────────────

# Excitation profile
MODE      = "pulse"
T_TOTAL   = 200.0
T_ON      = 100.0
T_OFF     = 3.0
N_PULSES  = 3

# TTA model
N_MED      = 4
KH_LIST    = [0.01, 1, 10.0, 100.0]
KDECAY_TTA = 1e-5
F_SPIN     = 1.0

# k_ex sweep (logspace)
KEX_MIN   = 1e-6
KEX_MAX   = 1e2
N_POINTS  = 100

UNITS     = "ns"


# ── Sweep ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kex_range = np.logspace(np.log10(KEX_MIN), np.log10(KEX_MAX), N_POINTS)

    n_kh = len(KH_LIST)
    total_runs = 2 * N_POINTS * n_kh
    print(f"TTA sweep: {N_POINTS} k_ex points × {n_kh} k_h values × 2 (homo/hetero) = {total_runs} runs")
    print(f"  mode={MODE}  T={T_TOTAL}  t_on={T_ON}  t_off={T_OFF}  n_pulses={N_PULSES}")
    print(f"  N={N_MED}  k_h={KH_LIST}  k_decay={KDECAY_TTA}  f_spin={F_SPIN}")

    total_photons      = {kh: np.zeros(N_POINTS) for kh in KH_LIST}
    total_photons_homo = {kh: np.zeros(N_POINTS) for kh in KH_LIST}

    pbar = tqdm(total=total_runs, desc="sweep", unit="run")
    for kh in KH_LIST:
        for i, k_ex in enumerate(kex_range):
            t0 = time.time()
            _, k_emit_t, dt = run_tta(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                      N_MED, kh, k_ex, KDECAY_TTA,
                                      False, F_SPIN)
            total_photons[kh][i] = float(np.sum(k_emit_t) * dt)
            t_hetero = time.time() - t0
            pbar.update(1)

            t0 = time.time()
            _, k_emit_t, dt = run_tta(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                      N_MED, kh, k_ex, KDECAY_TTA,
                                      True, F_SPIN)
            total_photons_homo[kh][i] = float(np.sum(k_emit_t) * dt)
            t_homo = time.time() - t0
            pbar.update(1)

            tqdm.write(f"  k_h={kh:g}  k_ex={k_ex:.3e}  hetero {t_hetero:.2f}s  homo {t_homo:.2f}s")
    pbar.close()

    params = dict(
        T_total=T_TOTAL,
        mode=MODE,
        t_on=T_ON,
        t_off=T_OFF,
        n_pulses=N_PULSES,
        n_med=N_MED,
        k_decay=KDECAY_TTA,
        f_spin=F_SPIN,
        kex_min=KEX_MIN,
        kex_max=KEX_MAX,
        n_points=N_POINTS,
        units=UNITS,
    )

    path = save_sweep("tta", kex_range, total_photons, params,
                      total_photons_homo=total_photons_homo)
    print(f"\nSaved: {path}")
