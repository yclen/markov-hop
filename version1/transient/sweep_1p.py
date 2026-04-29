import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from data_io import save_sweep
from transient_pulse import run_1p


# ── Parameters ────────────────────────────────────────────────────────────────

# Excitation profile
MODE      = "pulse"
T_TOTAL   = 20.0
T_ON      = 5.0
T_OFF     = 3.0
N_PULSES  = 3

# 1p model
KFLUOR_1P = 0.5

# k_ex sweep (logspace)
KEX_MIN   = 1e-3
KEX_MAX   = 1e2
N_POINTS  = 50

UNITS     = "ns"


# ── Sweep ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kex_range = np.logspace(np.log10(KEX_MIN), np.log10(KEX_MAX), N_POINTS)

    print(f"1p sweep: {N_POINTS} points, k_ex in [{KEX_MIN:.2e}, {KEX_MAX:.2e}] {UNITS}^-1")
    print(f"  mode={MODE}  T={T_TOTAL}  t_on={T_ON}  t_off={T_OFF}  n_pulses={N_PULSES}")
    print(f"  k_fluor={KFLUOR_1P}")

    total_photons = np.zeros(N_POINTS)
    for i, k_ex in enumerate(tqdm(kex_range, desc="k_ex", unit="pt")):
        _, k_emit_t, dt = run_1p(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                 k_ex, KFLUOR_1P)
        total_photons[i] = float(np.sum(k_emit_t) * dt)

    params = dict(
        T_total=T_TOTAL,
        mode=MODE,
        t_on=T_ON,
        t_off=T_OFF,
        n_pulses=N_PULSES,
        k_fluor=KFLUOR_1P,
        kex_min=KEX_MIN,
        kex_max=KEX_MAX,
        n_points=N_POINTS,
        units=UNITS,
    )

    path = save_sweep("1p", kex_range, total_photons, params)
    print(f"\nSaved: {path}")
