import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'steady-state'))
from data_io import get_file, load_sweep
from ss_tta import sweep_tta


# ── File ──────────────────────────────────────────────────────────────────────

MODEL = "tta"
DATE  = None   # None → today
N     = 0      # 0 = most recent

# ── Plot control ──────────────────────────────────────────────────────────────

KH_PLOT   = [1]   # None → all k_h values in file; e.g. [1.0, 100.0] to subset
SHOW_HOMO = True   # include dashed homo lines
SHOW_SS   = True   # include steady-state reference lines

# ── Style ─────────────────────────────────────────────────────────────────────

SS_COLOR         = "#1565C0"                                    # blue for all SS lines
TRANSIENT_COLORS = ["#E53935", "#43A047", "#8E24AA", "#FB8C00", "#00838F"]
LW               = 0.9


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_load(result):
    """Return (kex, photons_dict, photons_homo_dict | None, params)."""
    if len(result) == 4:
        kex, photons, photons_homo, params = result
    else:
        kex, photons, params = result
        photons_homo = None

    if not isinstance(photons, dict):
        kh = params.get("k_h", "?")
        photons = {kh: photons}
        if photons_homo is not None:
            photons_homo = {kh: photons_homo}

    return kex, photons, photons_homo, params


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = get_file(n=N, model=MODEL, date=DATE)
    kex, photons, photons_homo, params = _normalize_load(load_sweep(path))

    print(f"Loaded: {path}")
    print(f"  k_ex  : {len(kex)} points  {kex.min():.2e} → {kex.max():.2e} {params.get('units', '')}^-1")
    print(f"  k_h   : {list(photons.keys())}")
    print(f"  homo  : {photons_homo is not None}")
    print("  params:")
    for k, v in params.items():
        if k not in ("model", "timestamp", "k_h_values", "kex_min", "kex_max", "n_points", "units"):
            print(f"    {k} = {v}")

    # Apply k_h filter
    kh_list = [kh for kh in photons if KH_PLOT is None or kh in KH_PLOT]
    if not kh_list:
        raise ValueError(f"KH_PLOT={KH_PLOT} matched none of the available k_h: {list(photons)}")
    print(f"  Plotting k_h: {kh_list}  |  homo={SHOW_HOMO}  SS={SHOW_SS}")

    t_on    = params["t_on"]
    T_total = params["T_total"]

    kex_to_I        = 1e6
    per_ns_to_per_s = 1e9

    I_avg = kex_to_I * kex * t_on / T_total
    I_ss  = kex_to_I * kex

    kh_str = ", ".join(f"{kh:g}" for kh in kh_list)
    mode = params.get('mode')
    title = f"{MODEL}  |  mode={mode}  T={params.get('T_total')}  t_on={params.get('t_on')}"
    if mode == "pulse_train":
        title += f"  t_off={params.get('t_off')}  n_pulses={params.get('n_pulses')}"
    title += f"  N={params.get('n_med')}  k_h=[{kh_str}]"

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, kh in enumerate(kh_list):
        tc = TRANSIENT_COLORS[i % len(TRANSIENT_COLORS)]

        # Transient hetero
        cps_hetero = photons[kh] / T_total * per_ns_to_per_s
        ax.plot(I_avg, cps_hetero, color=tc, lw=LW, ls="-",
                label=f"transient  k_h={kh:g}")

        # Transient homo
        if SHOW_HOMO and photons_homo is not None:
            cps_homo = photons_homo[kh] / T_total * per_ns_to_per_s
            ax.plot(I_avg, cps_homo, color=tc, lw=LW, ls="--",
                    label=f"transient  k_h={kh:g}  (homo)")

        # Steady-state
        if SHOW_SS:
            print(f"  SS  k_h={kh:g}  hetero ...")
            _, k_emit_ss, _ = sweep_tta(kex_range=kex,
                                        n_med=params["n_med"], k_h=kh,
                                        k_decay=params["k_decay"], homo_tta=False,
                                        f_spin=params["f_spin"])
            ax.plot(I_ss, k_emit_ss * per_ns_to_per_s, color=SS_COLOR, lw=LW, ls="-",
                    label=f"SS  k_h={kh:g}")

            if SHOW_HOMO and photons_homo is not None:
                print(f"  SS  k_h={kh:g}  homo ...")
                _, k_emit_ss_homo, _ = sweep_tta(kex_range=kex,
                                                 n_med=params["n_med"], k_h=kh,
                                                 k_decay=params["k_decay"], homo_tta=True,
                                                 f_spin=params["f_spin"])
                ax.plot(I_ss, k_emit_ss_homo * per_ns_to_per_s, color=SS_COLOR, lw=LW, ls="--",
                        label=f"SS  k_h={kh:g}  (homo)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity  (W/cm$^2$)")
    ax.set_ylabel("cps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=9)

    plt.show()
