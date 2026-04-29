"""I/O helpers for transient k_ex sweeps.

Sweep results are saved as a single CSV per run:

    version1/data/YYYY-MM-DD/sweep_<model>_<HHMMSS>.csv

The CSV starts with `# key = value` comment lines holding every metadata
field (model, pulse settings, all model parameters), followed by two
data columns: k_ex and total_photons.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Save ─────────────────────────────────────────────────────────────────────

def save_sweep(model: str,
               kex_values: np.ndarray,
               total_photons: np.ndarray,
               params: dict,
               base_dir: Path | None = None,
               total_photons_homo: np.ndarray | None = None) -> Path:
    """Save a k_ex sweep to a timestamped CSV under data/YYYY-MM-DD/.

    Parameters
    ----------
    model              : "1p" | "2p" | "tta" — used in the filename and
                         stamped into the header.
    kex_values         : array of k_ex values.
    total_photons      : array of total photon counts (same length as
                         kex_values). For TTA, this is the homo=False rate.
    params             : dict of run parameters; every entry becomes a
                         `# key = value` header line.
    base_dir           : root data directory (defaults to version1/data/).
    total_photons_homo : optional array of total photon counts with homoTTA
                         enabled. If provided, a third column is written.

    Returns
    -------
    Path to the written CSV.
    """
    kex_values = np.asarray(kex_values)
    total_photons = np.asarray(total_photons)
    if kex_values.shape != total_photons.shape:
        raise ValueError(
            f"kex_values and total_photons must have the same shape "
            f"(got {kex_values.shape} vs {total_photons.shape})"
        )
    if total_photons_homo is not None:
        total_photons_homo = np.asarray(total_photons_homo)
        if total_photons_homo.shape != kex_values.shape:
            raise ValueError(
                f"total_photons_homo shape {total_photons_homo.shape} "
                f"doesn't match kex_values {kex_values.shape}"
            )

    base_dir = Path(base_dir) if base_dir else DEFAULT_DATA_DIR
    now = datetime.now()
    date_dir = base_dir / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = f"sweep_{model}_{now.strftime('%H%M%S')}.csv"
    path = date_dir / filename

    header = {"model": model, "timestamp": now.isoformat(timespec="seconds")}
    header.update(params)

    with path.open("w", newline="") as f:
        for key, value in header.items():
            f.write(f"# {key} = {value}\n")
        writer = csv.writer(f)
        if total_photons_homo is None:
            writer.writerow(["k_ex", "total_photons"])
            for kex, n in zip(kex_values, total_photons):
                writer.writerow([f"{kex:.6e}", f"{n:.6e}"])
        else:
            writer.writerow(["k_ex", "total_photons", "total_photons_homo"])
            for kex, n, nh in zip(kex_values, total_photons, total_photons_homo):
                writer.writerow([f"{kex:.6e}", f"{n:.6e}", f"{nh:.6e}"])

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
    model    : "1p" | "2p" | "tta" — filters files by `sweep_<model>_*.csv`.
    date     : date folder in "YYYY-MM-DD" format. Defaults to today.
    base_dir : root data directory (defaults to version1/data/).

    Returns
    -------
    Path to the matching CSV.
    """
    base_dir = Path(base_dir) if base_dir else DEFAULT_DATA_DIR
    date = date if date else datetime.now().strftime("%Y-%m-%d")
    date_dir = base_dir / date

    if not date_dir.is_dir():
        raise FileNotFoundError(f"No data folder for {date}: {date_dir}")

    matches = sorted(date_dir.glob(f"sweep_{model}_*.csv"), reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No sweep_{model}_*.csv files in {date_dir}"
        )
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
    For 2-column files (1p, 2p, or TTA without homo):
        (kex_values, total_photons, params)

    For 3-column files (TTA with homo column):
        (kex_values, total_photons, total_photons_homo, params)

    `params` is a dict parsed from the `# key = value` header lines, with
    light type coercion (int → float → bool → str).
    """
    path = Path(path)
    params: dict = {}
    column_line = ""

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
    total_photons = data[:, 1]

    if len(columns) == 3 and columns[2] == "total_photons_homo":
        total_photons_homo = data[:, 2]
        return kex_values, total_photons, total_photons_homo, params
    return kex_values, total_photons, params


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
