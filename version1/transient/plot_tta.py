import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'steady-state'))
from data_io import get_file, load_sweep
from ss_tta import sweep_tta


# ── Parameters ────────────────────────────────────────────────────────────────

MODEL = "tta"
DATE  = "2026-04-28" # None → today
N     = 0      # 0 = most recent


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = get_file(n=N, model=MODEL, date=DATE)
    kex, total_photons, total_photons_homo, params = load_sweep(path)

    print(f"Loaded: {path}")
    print(f"  {len(kex)} points  |  k_ex {kex.min():.2e} → {kex.max():.2e} {params.get('units', '')}^-1")

    # Exposed params for downstream use
    t_on    = params["t_on"]
    T_total = params["T_total"]

    # Unit conversions
    kex_to_I = 1e6           # ns^-1 → W/cm^2
    per_ns_to_per_s = 1e9    # ns^-1 → s^-1
    cps_no_homo   = total_photons      / T_total * per_ns_to_per_s
    cps_with_homo = total_photons_homo / T_total * per_ns_to_per_s
    I_peak = kex_to_I * kex
    I_avg  = I_peak * t_on / T_total

    units = params.get("units", "ns")
    title = (f"{MODEL}  |  mode={params.get('mode')}  "
             f"T={params.get('T_total')}  "
             f"t_on={params.get('t_on')}  t_off={params.get('t_off')}  "
             f"n_pulses={params.get('n_pulses')}  "
             f"N={params.get('n_med')}  k_h={params.get('k_h')}")

    # ── Steady-state sweeps on same k_ex range ─────────────────────────────
    print("Running ss_tta on same k_ex range — homo=False ...")
    _, k_emit_ss, _ = sweep_tta(kex_range=kex,
                                n_med=params["n_med"], k_h=params["k_h"],
                                k_decay=params["k_decay"], homo_tta=False,
                                f_spin=params["f_spin"])
    print("Running ss_tta — homo=True ...")
    _, k_emit_ss_homo, _ = sweep_tta(kex_range=kex,
                                     n_med=params["n_med"], k_h=params["k_h"],
                                     k_decay=params["k_decay"], homo_tta=True,
                                     f_spin=params["f_spin"])
    I_ss = kex_to_I * kex
    k_emit_ss      = k_emit_ss      * per_ns_to_per_s
    k_emit_ss_homo = k_emit_ss_homo * per_ns_to_per_s

    # ── Overlay plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(I_avg, cps_no_homo,   color="#2E86C1", linewidth=1.5,
            marker="o", markersize=3, label=f"{MODEL} transient (no homo)")
    ax.plot(I_avg, cps_with_homo, color="#2E86C1", linewidth=1.5, linestyle="--",
            marker="o", markersize=3, label=f"{MODEL} transient (homo)")
    ax.plot(I_ss,  k_emit_ss,      color="#C0392B", linewidth=1.5,
            marker="o", markersize=3, label=f"{MODEL} steady-state (no homo)")
    ax.plot(I_ss,  k_emit_ss_homo, color="#C0392B", linewidth=1.5, linestyle="--",
            marker="o", markersize=3, label=f"{MODEL} steady-state (homo)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity  (W/cm$^2$)")
    ax.set_ylabel("cps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=10)

    plt.show()
