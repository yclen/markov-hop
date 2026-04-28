import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from models import compute_2p_rates, G_FACTOR


def sweep_2p(kex_range, k_1, k_fluor, k_ex2=None, g_factor=G_FACTOR):
    """
    Steady-state sweep for the 2-photon sequential absorption model.

    Parameters
    ----------
    kex_range : array-like  Excitation rates to sweep over.
    k_1       : float       S1 decay rate.
    k_fluor   : float       Fluorescence rate.
    k_ex2     : float|None  S1 → upper state excitation rate.
                            If None, k_ex2 tracks k_ex at each point.
    g_factor  : float       Pulsed enhancement factor (default G_FACTOR). Pass 1.0 for CW.

    Returns
    -------
    kex    : np.ndarray  Excitation rate values.
    k_emit : np.ndarray  Emission rates.
    """
    kex_range = np.asarray(kex_range)
    k_emit    = np.zeros(len(kex_range))

    for i, k_ex in enumerate(tqdm(kex_range, desc="2p sweep", leave=False)):
        k_ex2_val = k_ex if k_ex2 is None else k_ex2
        k_emit[i] = compute_2p_rates(k_ex=k_ex, k_ex2=k_ex2_val, k_1=k_1,
                                     k_fluor=k_fluor, g_factor=g_factor)

    return kex_range, k_emit
