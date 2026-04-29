import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from models import build_1p_matrix, build_2p_matrix, build_tta_matrix


# ── Parameters ────────────────────────────────────────────────────────────────

# Numerics
P_MAX    = 0.1       # max probability per step — each model picks dt = P_MAX / max(rates)
T_TOTAL  = 20.0      # total simulation time
UNITS    = "ns"

# Excitation profile
MODE      = "pulse"   # "CW" | "pulse" | "pulse_train"
T_ON      = 10
T_OFF     = 3.0
N_PULSES  = 3

# 1p model
KEX_1P    = 1.0
KFLUOR_1P = .5

# 2p model  (g_factor = 1 in transient mode — pulse profile carries the time dependence)
g = 1e1
KEX_2P    = 1.0 * g
KEX2_2P   = 1.0 * g       # second-photon excitation rate (gates with the laser via M_on/M_off)
K1_2P     = 1e3       # S1 decay rate
KFLUOR_2P = KFLUOR_1P

# TTA model
N_MED      = 4
KH_TTA     = 10.0
KEX_TTA    = 1.0
KDECAY_TTA = 0.01
HOMO_TTA   = True
F_SPIN     = 1


# ── Excitation profile ────────────────────────────────────────────────────────

def _auto_dt(*rates):
    """Pick dt so the fastest rate has probability P_MAX per step."""
    return P_MAX / max(rates)


def make_excitation_profile(mode, T_total, dt, t_on, t_off, n_pulses):
    """Returns a 0/1 mask of length n_steps marking when the laser is ON."""
    n_steps = int(round(T_total / dt))
    profile = np.zeros(n_steps, dtype=np.int8)

    if mode == "CW":
        profile[:] = 1
    elif mode == "pulse":
        on_steps = int(round(t_on / dt))
        profile[:on_steps] = 1
    elif mode == "pulse_train":
        on_steps  = int(round(t_on  / dt))
        off_steps = int(round(t_off / dt))
        cycle = on_steps + off_steps
        for p in range(n_pulses):
            start = p * cycle
            end   = min(start + on_steps, n_steps)
            if start >= n_steps:
                break
            profile[start:end] = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'.")
    return profile


def pulse_boundaries(mode, T_total, t_on, t_off, n_pulses):
    """Times at which the laser switches on or off."""
    boundaries = []
    if mode == "pulse":
        boundaries.append(t_on)
    elif mode == "pulse_train":
        cycle = t_on + t_off
        for p in range(n_pulses):
            boundaries.append(p * cycle + t_on)
            boundaries.append((p + 1) * cycle)
    return [t for t in boundaries if t < T_total]


# ── Shared stepper ────────────────────────────────────────────────────────────

def _step_through(M_on, M_off, emit_mask, profile, dt, n_states):
    """Step state vector through the laser profile, recording per-step emission rate."""
    M_on  = sp.csc_matrix(M_on)
    M_off = sp.csc_matrix(M_off)

    assert np.allclose(M_on.sum(axis=0),  1.0), "M_on columns don't sum to 1"
    assert np.allclose(M_off.sum(axis=0), 1.0), "M_off columns don't sum to 1"

    state = np.zeros(n_states)
    state[0] = 1.0   # start fully in ground/empty state

    n_steps = len(profile)
    k_emit_t = np.zeros(n_steps)
    for step in range(n_steps):
        M = M_on if profile[step] else M_off
        k_emit_t[step] = (emit_mask @ state) / dt
        state = M.dot(state)
    return k_emit_t


# ── Per-model runners ─────────────────────────────────────────────────────────

def run_1p(mode, T_total, t_on, t_off, n_pulses, k_ex, k_fluor):
    dt       = _auto_dt(k_ex, k_fluor)
    profile  = make_excitation_profile(mode, T_total, dt, t_on, t_off, n_pulses)
    times    = np.arange(len(profile)) * dt

    Pex    = k_ex    * dt
    Pfluor = k_fluor * dt
    M_on,  emit_mask = build_1p_matrix(Pex=Pex, Pfluor=Pfluor)
    M_off, _         = build_1p_matrix(Pex=0.0, Pfluor=Pfluor)
    k_emit_t = _step_through(M_on, M_off, emit_mask, profile, dt, n_states=2)
    return times, k_emit_t, dt


def run_2p(mode, T_total, t_on, t_off, n_pulses, k_ex, k_ex2, k_1, k_fluor):
    dt       = _auto_dt(k_ex, k_ex2, k_1, k_fluor)
    profile  = make_excitation_profile(mode, T_total, dt, t_on, t_off, n_pulses)
    times    = np.arange(len(profile)) * dt

    Pex    = k_ex    * dt
    Pex2   = k_ex2   * dt
    P1     = k_1     * dt
    Pfluor = k_fluor * dt
    M_on,  emit_mask = build_2p_matrix(Pex=Pex,  Pex2=Pex2, P1=P1, Pfluor=Pfluor)
    M_off, _         = build_2p_matrix(Pex=0.0,  Pex2=0.0, P1=P1, Pfluor=Pfluor)
    k_emit_t = _step_through(M_on, M_off, emit_mask, profile, dt, n_states=3)
    return times, k_emit_t, dt


def run_tta(mode, T_total, t_on, t_off, n_pulses,
            n_med, k_h, k_ex, k_decay, homo_tta, f_spin):
    dt       = _auto_dt(k_h, k_ex, k_decay)
    profile  = make_excitation_profile(mode, T_total, dt, t_on, t_off, n_pulses)
    times    = np.arange(len(profile)) * dt

    Ph     = k_h     * dt
    Pex    = k_ex    * dt
    Pdecay = k_decay * dt
    M_on,  emit_mask, _ = build_tta_matrix(n_med=n_med, Ph=Ph, Pex=Pex,
                                            Pdecay=Pdecay, homo_tta=homo_tta, f_spin=f_spin)
    M_off, _,         _ = build_tta_matrix(n_med=n_med, Ph=Ph, Pex=0.0,
                                            Pdecay=Pdecay, homo_tta=homo_tta, f_spin=f_spin)
    n_states = 2 ** (n_med + 2)
    k_emit_t = _step_through(M_on, M_off, emit_mask, profile, dt, n_states)
    return times, k_emit_t, dt


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    boundaries = pulse_boundaries(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES)

    print("Running 1p ...")
    times_1p,  k_emit_1p,  dt_1p  = run_1p(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                           KEX_1P, KFLUOR_1P)
    print(f"  dt = {dt_1p:.2e} {UNITS}  ({len(times_1p)} steps)")

    print("Running 2p ...")
    times_2p,  k_emit_2p,  dt_2p  = run_2p(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                           KEX_2P, KEX2_2P, K1_2P, KFLUOR_2P)
    print(f"  dt = {dt_2p:.2e} {UNITS}  ({len(times_2p)} steps)")

    print("Running TTA ...")
    times_tta, k_emit_tta, dt_tta = run_tta(MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
                                            N_MED, KH_TTA, KEX_TTA, KDECAY_TTA,
                                            HOMO_TTA, F_SPIN)
    print(f"  dt = {dt_tta:.2e} {UNITS}  ({len(times_tta)} steps)")

    # If homo_tta is on, also run hetero-only for direct comparison
    if HOMO_TTA:
        print("Running TTA (no homo, for comparison) ...")
        times_tta_no, k_emit_tta_no, dt_tta_no = run_tta(
            MODE, T_TOTAL, T_ON, T_OFF, N_PULSES,
            N_MED, KH_TTA, KEX_TTA, KDECAY_TTA, False, F_SPIN,
        )
        print(f"  dt = {dt_tta_no:.2e} {UNITS}  ({len(times_tta_no)} steps)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Shade laser-ON regions (dt-independent — uses physical times only)
    if MODE == "CW":
        ax.axvspan(0.0, T_TOTAL, color="gold", alpha=0.10)
    elif MODE == "pulse":
        ax.axvspan(0.0, T_ON, color="gold", alpha=0.10)
    elif MODE == "pulse_train":
        cycle = T_ON + T_OFF
        for p in range(N_PULSES):
            start = p * cycle
            end   = min(start + T_ON, T_TOTAL)
            if start >= T_TOTAL:
                break
            ax.axvspan(start, end, color="gold", alpha=0.10)

    # Emission rates (1p blue, 2p red, tta orange)
    cum_1p  = float(np.sum(k_emit_1p))  * dt_1p
    cum_2p  = float(np.sum(k_emit_2p))  * dt_2p
    cum_tta = float(np.sum(k_emit_tta)) * dt_tta

    ax.plot(times_1p,  k_emit_1p,  color="#2E86C1", linewidth=1.5,
            label=f"1p   (cumul. = {cum_1p:.3f})")
    ax.plot(times_2p,  k_emit_2p,  color="#C0392B", linewidth=1.5,
            label=f"2p   (cumul. = {cum_2p:.3f})")

    if HOMO_TTA:
        cum_tta_no = float(np.sum(k_emit_tta_no)) * dt_tta_no
        ax.plot(times_tta_no, k_emit_tta_no, color="#E87722", linewidth=1.5,
                label=f"TTA hetero only  (cumul. = {cum_tta_no:.3f})")
        ax.plot(times_tta,    k_emit_tta,    color="#E87722", linewidth=1.5, linestyle="--",
                label=f"TTA + homo       (cumul. = {cum_tta:.3f})")
    else:
        ax.plot(times_tta, k_emit_tta, color="#E87722", linewidth=1.5,
                label=f"TTA  (cumul. = {cum_tta:.3f})")

    for t in boundaries:
        ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(f"Time  ({UNITS})")
    ax.set_ylabel(f"Emission rate  ({UNITS}$^{{-1}}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    title = (f"Transient emission  |  mode={MODE}  |  N={N_MED}  "
             f"$k_h$={KH_TTA}  $k_{{ex}}$(tta)={KEX_TTA}  $k_{{decay}}$={KDECAY_TTA}  "
             f"f_spin={F_SPIN}  |  homo_tta={HOMO_TTA}  |  P_max={P_MAX}")
    ax.set_title(title, fontsize=10)

    plt.show()
