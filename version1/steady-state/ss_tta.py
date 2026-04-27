import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from models import compute_tta_rates


def sweep_tta(kex_range, n_med, k_h, k_decay, homo_tta=False, f_spin=0.4):
    """
    Steady-state sweep for the TTA-UC chain model.

    Parameters
    ----------
    kex_range : array-like  Excitation rates to sweep over.
    n_med     : int         Number of mediator sites.
    k_h       : float       Hopping rate.
    k_decay   : float       Decay rate per occupied site.
    homo_tta  : bool        Enable mediator-mediator homoTTA.
    f_spin    : float       Spin-statistical factor (default 0.4).

    Returns
    -------
    kex    : np.ndarray  Excitation rate values.
    k_emit : np.ndarray  HeteroTTA emission rates.
    k_homo : np.ndarray  HomoTTA annihilation rates (zeros if homo_tta=False).
    """
    kex_range = np.asarray(kex_range)
    k_emit    = np.zeros(len(kex_range))
    k_homo    = np.zeros(len(kex_range))

    for i, k_ex in enumerate(tqdm(kex_range, desc=f"TTA sweep (N={n_med})", leave=False)):
        k_emit[i], k_homo[i] = compute_tta_rates(
            n_med=n_med, k_h=k_h, k_ex=k_ex, k_decay=k_decay,
            homo_tta=homo_tta, f_spin=f_spin,
        )

    return kex_range, k_emit, k_homo
