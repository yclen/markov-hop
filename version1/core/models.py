import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs

P_MAX = 0.1   # max probability per step — controls numerical stability


# ── TTA Model ────────────────────────────────────────────────────────────────
#
# State encoding: (n_med + 2)-bit integer
#   bit (n_med+1) = sensitizer (S)
#   bit n_med..1  = mediators M1..MN
#   bit 0         = annihilator (A)
#
# Transitions:
#   S vacant  → excited with Pex
#   S occupied → hop right to M1 (if vacant); homoTTA if occupied and homo_tta
#   M1        → right hop only (no back-hop onto S)
#   M2..MN-1  → bidirectional hops
#   MN        → left hop; right hop onto A: heteroTTA if A occupied, else normal
#   Decay     → any occupied site clears with Pdecay

def build_tta_matrix(n_med, Ph, Pex, Pdecay=0.0, homo_tta=False, f_spin=0.4):
    """
    Build the TTA Markov transition matrix.

    Parameters
    ----------
    n_med    : int    Number of mediator sites.
    Ph       : float  Hopping probability per step.
    Pex      : float  Excitation probability per step on S.
    Pdecay   : float  Decay probability per step per occupied site.
    homo_tta : bool   Mediator-mediator annihilation on collision.
    f_spin   : float  Spin-statistical factor (default 0.4). Fraction of heteroTTA
                      collisions that produce a photon. The remaining (1-f_spin)
                      fraction quenches MN but leaves A occupied.

    Returns
    -------
    M         : np.ndarray (n_states, n_states)  Column-stochastic transition matrix.
    emit_mask : np.ndarray (n_states,)           HeteroTTA emission probability per state per step.
    homo_mask : np.ndarray (n_states,)           HomoTTA annihilation probability per state per step.
    """
    n_sites  = n_med + 2
    n_states = 2 ** n_sites
    s_idx    = 0
    a_idx    = n_med + 1

    def occ(state, site):
        return (state >> (n_sites - 1 - site)) & 1

    def flip(state, site):
        return state ^ (1 << (n_sites - 1 - site))

    M         = np.zeros((n_states, n_states))
    emit_mask = np.zeros(n_states)
    homo_mask = np.zeros(n_states)

    for s in range(n_states):
        trans = {}

        def add(t, p):
            trans[t] = trans.get(t, 0.0) + p

        # sensitizer
        if occ(s, s_idx) == 0:
            add(flip(s, s_idx), Pex)
        else:
            if occ(s, 1) == 0:
                add(flip(flip(s, s_idx), 1), Ph)
            elif homo_tta:
                add(flip(flip(s, s_idx), 1), Ph)
                homo_mask[s] += Ph

        # mediators
        for m in range(1, n_med + 1):
            if occ(s, m) == 0:
                continue

            is_first = (m == 1)
            is_last  = (m == n_med)

            if not is_first:
                left = m - 1
                if occ(s, left) == 0:
                    add(flip(flip(s, m), left), Ph)
                elif homo_tta:
                    add(flip(flip(s, m), left), Ph)
                    homo_mask[s] += Ph

            if is_last:
                if occ(s, a_idx) == 0:
                    add(flip(flip(s, m), a_idx), Ph)
                else:
                    # successful TTA: both MN and A clear, photon emitted
                    add(flip(flip(s, m), a_idx), Ph * f_spin)
                    emit_mask[s] = Ph * f_spin
                    # failed TTA: only MN clears, A stays occupied, no photon
                    add(flip(s, m), Ph * (1 - f_spin))
            else:
                right = m + 1
                if occ(s, right) == 0:
                    add(flip(flip(s, m), right), Ph)
                elif homo_tta:
                    add(flip(flip(s, m), right), Ph)
                    homo_mask[s] += Ph

        # decay
        if Pdecay > 0.0:
            for site in range(n_sites):
                if occ(s, site) == 1:
                    add(flip(s, site), Pdecay)

        # normalise column
        total_out = sum(p for t, p in trans.items() if t != s)
        if total_out > 1.0:
            scale = 1.0 / total_out
            trans = {t: p * scale for t, p in trans.items() if t != s}
            emit_mask[s] *= scale
            homo_mask[s] *= scale
            total_out = 1.0
        trans[s] = trans.get(s, 0.0) + (1.0 - total_out)

        for t, p in trans.items():
            M[t, s] += p

    return M, emit_mask, homo_mask


def _steady_state(M):
    """Sparse eigenvector solve for the eigenvalue-1 steady-state vector."""
    _, eigvec = eigs(csc_matrix(M), k=1, which="LM", sigma=1.0)
    steady = eigvec[:, 0].real
    steady /= steady.sum()
    return steady


def compute_tta_rates(n_med, k_h, k_ex, k_decay, homo_tta=False, f_spin=0.4):
    """
    Steady-state emission and homoTTA rates for the TTA-UC chain.

    dt is chosen automatically from P_MAX for numerical stability.

    Returns
    -------
    k_emit : float  HeteroTTA emission rate (ns⁻¹)
    k_homo : float  HomoTTA annihilation rate (ns⁻¹), 0 if homo_tta=False
    """
    dt     = P_MAX / max(k_h, k_ex, k_decay)
    Ph     = k_h     * dt
    Pex    = k_ex    * dt
    Pdecay = k_decay * dt

    M, emit_mask, homo_mask = build_tta_matrix(n_med=n_med, Ph=Ph, Pex=Pex,
                                               Pdecay=Pdecay, homo_tta=homo_tta,
                                               f_spin=f_spin)
    steady = _steady_state(M)
    k_emit = float(emit_mask @ steady) / dt
    k_homo = float(homo_mask @ steady) / dt if homo_tta else 0.0
    return k_emit, k_homo


# ── 1-Photon Model ───────────────────────────────────────────────────────────
#
# Two-state fluorophore: state 0 = ground, state 1 = excited.
# Emission rate saturates as k_emit → k_fluor for k_ex >> k_fluor  (linear regime).

def build_1p_matrix(Pex, Pfluor):
    M = np.array([
        [1 - Pex,    Pfluor   ],
        [Pex,     1 - Pfluor  ],
    ])
    emit_mask = np.array([0.0, Pfluor])
    return M, emit_mask


def compute_1p_rates(k_ex, k_fluor):
    """
    Steady-state emission rate for a simple two-state fluorophore.
    Analytical: k_emit = k_ex * k_fluor / (k_ex + k_fluor).

    Returns
    -------
    k_emit : float
    """
    k_emit = k_ex * k_fluor / (k_ex + k_fluor)
    return k_emit


# ── 2-Photon Model ───────────────────────────────────────────────────────────
#
# Three-state sequential two-photon absorption:
#   state 0 = ground, state 1 = intermediate S1, state 2 = upper emitting state.
#
# G_FACTOR accounts for pulsed-excitation I² enhancement relative to CW.
# For CW excitation pass g_factor=1.0.

PULSE_DURATION_NS = 1e-4          # 100 fs in ns
REP_RATE_NS       = 1 / 12.5      # 80 MHz → period in ns
G_FACTOR          = 1.0 / (PULSE_DURATION_NS * REP_RATE_NS)   # ≈ 125,000


def build_2p_matrix(Pex, Pex2, P1, Pfluor):
    M = np.array([
        [1 - Pex,        P1,          Pfluor ],
        [Pex,     1 - P1 - Pex2,      0.0    ],
        [0.0,            Pex2,     1 - Pfluor ],
    ])
    emit_mask = np.array([0.0, 0.0, Pfluor])
    return M, emit_mask


def compute_2p_rates(k_ex, k_ex2, k_1, k_fluor, g_factor=G_FACTOR):
    """
    Steady-state emission rate for sequential two-photon absorption.

    Parameters
    ----------
    k_ex      : float  Ground → S1 excitation rate.
    k_ex2     : float  S1 → upper state excitation rate.
    k_1       : float  S1 → ground decay rate.
    k_fluor   : float  Upper state fluorescence rate.
    g_factor  : float  Pulsed enhancement factor (default G_FACTOR ≈ 125,000).
                       Applied as a post-scaling on k_emit. Pass 1.0 for CW.

    Returns
    -------
    k_emit : float
    """
    dt     = P_MAX / max(k_ex, k_ex2, k_1, k_fluor)
    Pex    = k_ex    * dt
    Pex2   = k_ex2   * dt
    P1     = k_1     * dt
    Pfluor = k_fluor * dt

    M, emit_mask = build_2p_matrix(Pex, Pex2, P1, Pfluor)

    vals, vecs = np.linalg.eig(M)
    idx    = np.argmin(np.abs(vals - 1.0))
    steady = vecs[:, idx].real
    steady /= steady.sum()

    k_emit = float(emit_mask @ steady) / dt
    return k_emit * g_factor
