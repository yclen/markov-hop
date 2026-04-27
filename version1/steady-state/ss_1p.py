import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from models import compute_1p_rates


def sweep_1p(kex_range, k_fluor):
    """
    Steady-state sweep for the 1-photon fluorophore model.

    Parameters
    ----------
    kex_range : array-like  Excitation rates to sweep over.
    k_fluor   : float       Fluorescence rate.

    Returns
    -------
    kex    : np.ndarray  Excitation rate values.
    k_emit : np.ndarray  Emission rates.
    """
    kex_range = np.asarray(kex_range)
    k_emit    = np.zeros(len(kex_range))

    for i, k_ex in enumerate(tqdm(kex_range, desc="1p sweep", leave=False)):
        k_emit[i] = compute_1p_rates(k_ex=k_ex, k_fluor=k_fluor)

    return kex_range, k_emit
