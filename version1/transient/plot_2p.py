import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'steady-state'))
from data_io import get_file, load_sweep
from ss_2p import sweep_2p


# ── Parameters ────────────────────────────────────────────────────────────────

MODEL = "2p"
DATE  = "2026-04-28"   # None → today
N     = 0      # 0 = most recent


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = get_file(n=N, model=MODEL, date=DATE)
    kex, total_photons, params = load_sweep(path)

    print(f"Loaded: {path}")
    print(f"  {len(kex)} points  |  k_ex {kex.min():.2e} → {kex.max():.2e} {params.get('units', '')}^-1")

    # Exposed params for downstream use
    t_on    = params["t_on"]
    T_total = params["T_total"]

    # g-factor for SS comparison: 1 / duty_cycle = T_total / t_on
    g_factor = np.sqrt(T_total / t_on)

    #unit conversions
    kex_to_I = 1e6 # from ns^-1 to W/cm^2
    per_ns_to_per_s = 1e9 # from ns^-1 to s^-1
    cps = total_photons/T_total * per_ns_to_per_s
    I_peak = kex_to_I * kex
    I_avg = I_peak * t_on / T_total

    units = params.get("units", "ns")
    title = (f"{MODEL}  |  mode={params.get('mode')}  "
             f"T={params.get('T_total')}  "
             f"t_on={params.get('t_on')}  t_off={params.get('t_off')}  "
             f"n_pulses={params.get('n_pulses')}  g={g_factor:.1f}")

    # ── Steady-state sweep on same k_ex range ──────────────────────────────
    print(f"Running ss_2p on same k_ex range (g_factor={g_factor:.2f}) ...")
    kex_ss, k_emit_ss = sweep_2p(kex_range=kex, k_1=params["k_1"],
                                 k_fluor=params["k_fluor"], g_factor=g_factor)
    I_ss = kex_to_I * kex_ss
    k_emit_ss = k_emit_ss * per_ns_to_per_s

    # ── Overlay plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(I_avg, cps, color="#2E86C1", linewidth=1.5,
            marker="o", markersize=3, label=f"{MODEL} transient")
    ax.plot(I_ss, k_emit_ss, color="#C0392B", linewidth=1.5,
            marker="o", markersize=3, label=f"{MODEL} steady-state")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity  (W/cm$^2$)")
    ax.set_ylabel("cps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=10)

    plt.show()
