import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'steady-state'))
from data_io import get_file, load_sweep
from ss_1p import sweep_1p
from ss_2p import sweep_2p
from ss_tta import sweep_tta


# ── File selectors ────────────────────────────────────────────────────────────

DATE_1P  = None;  N_1P  = 0
DATE_2P  = None;  N_2P  = 0
DATE_TTA = None;  N_TTA = 0

# ── Plot control ──────────────────────────────────────────────────────────────

SHOW_1P           = True
SHOW_1P_TRANSIENT = True    # False → SS only, no file needed

SHOW_2P           = True
SHOW_2P_TRANSIENT = True    # False → SS only, no file needed

SHOW_TTA          = True

SHOW_SS           = True    # steady-state lines for all active models
SHOW_HOMO         = True    # TTA dashed homo lines
KH_PLOT           = [1]    # None → all TTA k_h values; e.g. [1.0, 10.0]

# ── Standalone SS params ──────────────────────────────────────────────────────
# Used for 1p / 2p when SHOW_Xp_TRANSIENT = False (no file loaded).
# Also used as fallback if a transient file is not found.

KEX_SS_MIN  = 1e-6
KEX_SS_MAX  = 1e2
KEX_SS_N    = 200

K_FLUOR_1P  = 0.5           # 1p fluorescence rate

K_1_2P      = 1e5           # 2p S1 decay rate
K_FLUOR_2P  = 0.5
G_FACTOR_2P = np.sqrt(1e4)          # None = sqrt(T_total/t_on) from file; float = explicit override

# ── Style ─────────────────────────────────────────────────────────────────────
# Each model has a (transient, SS) color pair — same hue, different shade.
# TTA cycles through pairs starting with brown (avoids blue/red conflict).

COLOR_1P = ("#90CAF9", "#1565C0")   # light blue / dark blue
COLOR_2P = ("#EF9A9A", "#B71C1C")   # light red  / dark red

TTA_COLORS = [
    ("#81C784", "#2E7D32"),   # green
    ("#A1887F", "#4E342E"),   # brown
    ("#CE93D8", "#6A1B9A"),   # purple
    ("#80CBC4", "#00695C"),   # teal
    ("#AED581", "#558B2F"),   # olive
]

LW = 0.9


# ── Helpers ───────────────────────────────────────────────────────────────────

_SKIP    = {"model", "timestamp", "k_h_values", "kex_min", "kex_max", "n_points", "units"}
_PER_NS  = 1e9
_KEX_TO_I = 1e6


def _print_file_info(label, path, kex, params, extra=None):
    print(f"\n{label}: {path}")
    print(f"  k_ex  : {len(kex)} points  {kex.min():.2e} → {kex.max():.2e} {params.get('units','')}^-1")
    if extra:
        for line in extra:
            print(f"  {line}")
    print("  params:")
    for k, v in params.items():
        if k not in _SKIP:
            print(f"    {k} = {v}")


def _normalize_tta(result):
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


def _I_avg(kex, params):
    return _KEX_TO_I * kex * params["t_on"] / params["T_total"]


def _I_peak(kex):
    return _KEX_TO_I * kex


def _cps(photons, T_total):
    return photons / T_total * _PER_NS


def _standalone_kex():
    return np.logspace(np.log10(KEX_SS_MIN), np.log10(KEX_SS_MAX), KEX_SS_N)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pulse_params = None   # filled by first successful transient file load

    # ── 1p ───────────────────────────────────────────────────────────────────
    if SHOW_1P:
        kex_ss, k_fluor_ss = None, None

        if SHOW_1P_TRANSIENT:
            try:
                path = get_file(n=N_1P, model="1p", date=DATE_1P)
                kex_1p, phot_1p, params_1p = load_sweep(path)
                _print_file_info("1p", path, kex_1p, params_1p)
                ax.plot(_I_avg(kex_1p, params_1p), _cps(phot_1p, params_1p["T_total"]),
                        color=COLOR_1P[0], lw=LW, ls="-", label="1p  pulsed")
                kex_ss     = kex_1p
                k_fluor_ss = params_1p["k_fluor"]
                if pulse_params is None:
                    pulse_params = params_1p
            except FileNotFoundError as e:
                print(f"[skip 1p transient] {e}")

        if SHOW_SS:
            if kex_ss is None:
                kex_ss     = _standalone_kex()
                k_fluor_ss = K_FLUOR_1P
                print(f"\n1p SS (standalone)  k_fluor={k_fluor_ss}"
                      f"  k_ex {kex_ss.min():.2e} → {kex_ss.max():.2e}")
            print("  Computing 1p SS ...")
            _, k_ss = sweep_1p(kex_ss, k_fluor=k_fluor_ss)
            ax.plot(_I_peak(kex_ss), k_ss * _PER_NS,
                    color=COLOR_1P[1], lw=LW, ls="-", label="1p  CW")

    # ── 2p ───────────────────────────────────────────────────────────────────
    if SHOW_2P:
        kex_ss, k_1_ss, k_fluor_ss, g_ss = None, None, None, None

        if SHOW_2P_TRANSIENT:
            try:
                path = get_file(n=N_2P, model="2p", date=DATE_2P)
                kex_2p, phot_2p, params_2p = load_sweep(path)
                _print_file_info("2p", path, kex_2p, params_2p)
                ax.plot(_I_avg(kex_2p, params_2p), _cps(phot_2p, params_2p["T_total"]),
                        color=COLOR_2P[0], lw=LW, ls="-", label="2p  pulsed")
                kex_ss     = kex_2p
                k_1_ss     = params_2p["k_1"]
                k_fluor_ss = params_2p["k_fluor"]
                g_ss = G_FACTOR_2P if G_FACTOR_2P is not None else np.sqrt(
                    params_2p["T_total"] / params_2p["t_on"])
                if pulse_params is None:
                    pulse_params = params_2p
            except FileNotFoundError as e:
                print(f"[skip 2p transient] {e}")

        if SHOW_SS:
            if kex_ss is None:
                kex_ss     = _standalone_kex()
                k_1_ss     = K_1_2P
                k_fluor_ss = K_FLUOR_2P
                g_ss       = G_FACTOR_2P if G_FACTOR_2P is not None else 1.0
                print(f"\n2p SS (standalone)  k_1={k_1_ss}  k_fluor={k_fluor_ss}"
                      f"  g={g_ss}  k_ex {kex_ss.min():.2e} → {kex_ss.max():.2e}")
            elif G_FACTOR_2P is not None:
                g_ss = G_FACTOR_2P
            print(f"  Computing 2p SS  (g_factor={g_ss:.3g}) ...")
            _, k_ss = sweep_2p(kex_ss, k_1=k_1_ss, k_fluor=k_fluor_ss, g_factor=g_ss)
            ax.plot(_I_peak(kex_ss), k_ss * _PER_NS,
                    color=COLOR_2P[1], lw=LW, ls="-", label=f"2p  CW  (g={g_ss:.2g})")

    # ── TTA ──────────────────────────────────────────────────────────────────
    if SHOW_TTA:
        try:
            path = get_file(n=N_TTA, model="tta", date=DATE_TTA)
            kex_tta, photons, photons_homo, params_tta = _normalize_tta(load_sweep(path))
            kh_list = list(photons.keys())
            _print_file_info("TTA", path, kex_tta, params_tta,
                             extra=[f"k_h   : {kh_list}",
                                    f"homo  : {photons_homo is not None}"])
            if pulse_params is None:
                pulse_params = params_tta

            kh_plot = [kh for kh in kh_list if KH_PLOT is None or kh in KH_PLOT]
            if not kh_plot:
                raise ValueError(f"KH_PLOT={KH_PLOT} matched none of available k_h: {kh_list}")

            for i, kh in enumerate(kh_plot):
                tc, tc_ss = TTA_COLORS[i % len(TTA_COLORS)]

                ax.plot(_I_avg(kex_tta, params_tta),
                        _cps(photons[kh], params_tta["T_total"]),
                        color=tc, lw=LW, ls="-", label=f"TTA  k_h={kh:g}  pulsed")

                if SHOW_HOMO and photons_homo is not None:
                    ax.plot(_I_avg(kex_tta, params_tta),
                            _cps(photons_homo[kh], params_tta["T_total"]),
                            color=tc, lw=LW, ls="--", label=f"TTA  k_h={kh:g}  pulsed  homoTTA")

                if SHOW_SS:
                    print(f"  SS TTA  k_h={kh:g}  hetero ...")
                    _, k_ss, _ = sweep_tta(kex_tta, n_med=params_tta["n_med"], k_h=kh,
                                           k_decay=params_tta["k_decay"], homo_tta=False,
                                           f_spin=params_tta["f_spin"])
                    ax.plot(_I_peak(kex_tta), k_ss * _PER_NS,
                            color=tc_ss, lw=LW, ls="-",
                            label=f"TTA  k_h={kh:g}  CW")

                    if SHOW_HOMO and photons_homo is not None:
                        print(f"  SS TTA  k_h={kh:g}  homo ...")
                        _, k_ss_homo, _ = sweep_tta(kex_tta, n_med=params_tta["n_med"], k_h=kh,
                                                     k_decay=params_tta["k_decay"], homo_tta=True,
                                                     f_spin=params_tta["f_spin"])
                        ax.plot(_I_peak(kex_tta), k_ss_homo * _PER_NS,
                                color=tc_ss, lw=LW, ls="--",
                                label=f"TTA  k_h={kh:g}  CW  homoTTA")

        except FileNotFoundError as e:
            print(f"[skip TTA] {e}")

    # ── Title ─────────────────────────────────────────────────────────────────
    title = "Model Comparison  |  Pulsed (light)  ·  CW (dark)  ·  homoTTA (dashed)"
    if pulse_params is not None:
        t_on    = pulse_params["t_on"]
        T_total = pulse_params["T_total"]
        rep_hz  = 1e9 / T_total
        title += f"\npulse width = {t_on:g} ns  ·  Rep Rate = {rep_hz:.3g} Hz"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity  (W/cm$^2$)")
    ax.set_ylabel("cps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=9)

    plt.show()
