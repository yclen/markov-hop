import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from data_io import save_sweep
from transient_pulse import run_tta


# ── Parameters ────────────────────────────────────────────────────────────────

# Excitation profile
MODE      = "pulse"
T_TOTAL   = 20.0
T_ON      = 20.0
T_OFF     = 3.0
N_PULSES  = 3

# TTA model
N_MED      = 4
KH_TTA     = 10.0
KDECAY_TTA = 0.01
F_SPIN     = 1.0

# k_ex sweep (logspace)
KEX_MIN   = 1e-3
KEX_MAX   = 1e2
N_POINTS  = 50

UNITS     = "ns"


# ── Sweep ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kex_range = np.logspace(np.log10(KEX_MIN), np.log10(KEX_MAX), N_POINTS)

    print(f"TTA sweep: {N_POINTS} points, k_ex in [{KEX_MIN:.2e}, {KEX_MAX:.2e}] {UNITS}^-1")
    print(f"  mode={MODE}  T={T_TOTAL}  t_on={T_ON}  t_off={T_OFF}  n_pulses={N_PULSES}")
    print(f"  N={N_MED}  k_h={KH_TTA}  k_decay={KDECAY_TTA}  f_spin={F_SPIN}")
    print(f"  Running both homo=False and homo=True per point ({2 * N_POINTS} total runs)")

    total_photons      = np.zeros(N_POINTS)
    total_photons_homo = np.zeros(N_POINTS)

    pbar = tqdm(total=2 * N_POINTS, desc="k_ex", unit="run")
    for i, k_ex in enumerate(kex_range):
        _, k_emit_t, dt = run_tta(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                  N_MED, KH_TTA, k_ex, KDECAY_TTA,
                                  False, F_SPIN)
        total_photons[i] = float(np.sum(k_emit_t) * dt)
        pbar.update(1)

        _, k_emit_t, dt = run_tta(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                  N_MED, KH_TTA, k_ex, KDECAY_TTA,
                                  True, F_SPIN)
        total_photons_homo[i] = float(np.sum(k_emit_t) * dt)
        pbar.update(1)
    pbar.close()

    params = dict(
        T_total=T_TOTAL,
        mode=MODE,
        t_on=T_ON,
        t_off=T_OFF,
        n_pulses=N_PULSES,
        n_med=N_MED,
        k_h=KH_TTA,
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
