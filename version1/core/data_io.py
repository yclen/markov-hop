"""I/O helpers for transient k_ex sweeps.

Sweep results are saved as a single CSV per run:

    version1/data/YYYY-MM-DD/sweep_<model>_<HHMMSS>.csv

The CSV starts with `# key = value` comment lines holding every metadata
field (model, pulse settings, all model parameters), followed by data columns.

Single-series files (1p, 2p, single-kh TTA):
    k_ex, total_photons [, total_photons_homo]

Multi-kh files (TTA with kh_list):
    k_ex, total_photons_kh1, total_photons_kh10, ...,
          total_photons_homo_kh1, total_photons_homo_kh10, ...
    The header also contains: # k_h_values = 1,10,100
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _kh_label(kh: float) -> str:
    return f"{kh:g}"


# ── Save ─────────────────────────────────────────────────────────────────────

def save_sweep(model: str,
               kex_values: np.ndarray,
               total_photons: np.ndarray | dict,
               params: dict,
               base_dir: Path | None = None,
               total_photons_homo: np.ndarray | dict | None = None) -> Path:
    """Save a k_ex sweep to a timestamped CSV under data/YYYY-MM-DD/.

    Parameters
    ----------
    model              : "1p" | "2p" | "tta"
    kex_values         : 1-D array of k_ex values.
    total_photons      : 1-D array (single k_h) or dict {k_h: array} (multi-kh).
    params             : run parameters written as `# key = value` header lines.
    base_dir           : root data directory (defaults to version1/data/).
    total_photons_homo : None, 1-D array, or dict {k_h: array} matching
                         total_photons.

    Returns
    -------
    Path to the written CSV.
    """
    kex_values = np.asarray(kex_values)
    multi_kh = isinstance(total_photons, dict)

    if multi_kh:
        photons = {kh: np.asarray(arr) for kh, arr in total_photons.items()}
    else:
        photons = {None: np.asarray(total_photons)}

    for kh, arr in photons.items():
        if arr.shape != kex_values.shape:
            raise ValueError(
                f"total_photons shape mismatch for k_h={kh}: "
                f"{arr.shape} vs kex_values {kex_values.shape}"
            )

    if total_photons_homo is not None:
        if isinstance(total_photons_homo, dict):
            photons_homo = {kh: np.asarray(arr) for kh, arr in total_photons_homo.items()}
        else:
            photons_homo = {None: np.asarray(total_photons_homo)}
        for kh, arr in photons_homo.items():
            if arr.shape != kex_values.shape:
                raise ValueError(
                    f"total_photons_homo shape mismatch for k_h={kh}: "
                    f"{arr.shape} vs kex_values {kex_values.shape}"
                )
    else:
        photons_homo = None

    base_dir = Path(base_dir) if base_dir else DEFAULT_DATA_DIR
    now = datetime.now()
    date_dir = base_dir / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = f"sweep_{model}_{now.strftime('%H%M%S')}.csv"
    path = date_dir / filename

    header = {"model": model, "timestamp": now.isoformat(timespec="seconds")}
    if multi_kh:
        header["k_h_values"] = ",".join(_kh_label(kh) for kh in photons)
    header.update(params)

    if multi_kh:
        hetero_cols = [f"total_photons_kh{_kh_label(kh)}" for kh in photons]
        homo_cols = (
            [f"total_photons_homo_kh{_kh_label(kh)}" for kh in photons_homo]
            if photons_homo else []
        )
    else:
        hetero_cols = ["total_photons"]
        homo_cols = ["total_photons_homo"] if photons_homo else []

    with path.open("w", newline="") as f:
        for key, value in header.items():
            f.write(f"# {key} = {value}\n")
        writer = csv.writer(f)
        writer.writerow(["k_ex"] + hetero_cols + homo_cols)

        hetero_arrays = list(photons.values())
        homo_arrays = list(photons_homo.values()) if photons_homo else []

        for i, kex in enumerate(kex_values):
            row = [f"{kex:.6e}"]
            row += [f"{arr[i]:.6e}" for arr in hetero_arrays]
            row += [f"{arr[i]:.6e}" for arr in homo_arrays]
            writer.writerow(row)

    return path


# ── Load ─────────────────────────────────────────────────────────────────────

def get_file(n: int = 0,
             model: str = "1p",
             date: str | None = None,
             base_dir: Path | None = None) -> Path:
    """Find a sweep CSV by model, date, and recency index.

    Parameters
    ----------
    n        : recency index — 0 returns the most recent file, 1 the next
               most recent, and so on.
    model    : "1p" | "2p" | "tta"
    date     : date folder in "YYYY-MM-DD" format. Defaults to today.
    base_dir : root data directory (defaults to version1/data/).
    """
    base_dir = Path(base_dir) if base_dir else DEFAULT_DATA_DIR
    date = date if date else datetime.now().strftime("%Y-%m-%d")
    date_dir = base_dir / date

    if not date_dir.is_dir():
        raise FileNotFoundError(f"No data folder for {date}: {date_dir}")

    matches = sorted(date_dir.glob(f"sweep_{model}_*.csv"), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No sweep_{model}_*.csv files in {date_dir}")
    if n >= len(matches):
        raise IndexError(
            f"Requested n={n} but only {len(matches)} sweep_{model} "
            f"files exist in {date_dir}"
        )
    return matches[n]


def load_sweep(path: Path):
    """Load a sweep CSV written by save_sweep.

    Returns
    -------
    Single-series (1p / 2p / old single-kh TTA without homo):
        (kex_values, total_photons, params)

    Single-series TTA with homo column:
        (kex_values, total_photons, total_photons_homo, params)

    Multi-kh TTA without homo:
        (kex_values, photons_dict, params)
        where photons_dict = {k_h_float: array, ...}

    Multi-kh TTA with homo columns:
        (kex_values, photons_dict, photons_homo_dict, params)
    """
    path = Path(path)
    params: dict = {}

    with path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition("=")
                params[key.strip()] = _coerce(value.strip())
            else:
                column_line = line.strip()
                break

    columns = [c.strip() for c in column_line.split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=len(params) + 1)
    kex_values = data[:, 0]

    hetero_cols = [c for c in columns if c.startswith("total_photons_kh")]
    homo_cols   = [c for c in columns if c.startswith("total_photons_homo_kh")]

    if hetero_cols:
        kh_values = [float(v) for v in str(params["k_h_values"]).split(",")]
        photons_dict = {
            kh: data[:, columns.index(col)]
            for kh, col in zip(kh_values, hetero_cols)
        }
        if homo_cols:
            photons_homo_dict = {
                kh: data[:, columns.index(col)]
                for kh, col in zip(kh_values, homo_cols)
            }
            return kex_values, photons_dict, photons_homo_dict, params
        return kex_values, photons_dict, params

    if "total_photons_homo" in columns:
        return kex_values, data[:, 1], data[:, 2], params

    return kex_values, data[:, 1], params


def _coerce(value: str):
    """Best-effort string → int/float/bool/str conversion."""
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
