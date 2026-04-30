# Claude's Notes — version1 changes

Changes made across sessions on 2026-04-29 and 2026-04-30.

---

## README.md

Brought the folder tree up to date with the actual state of the repo:

- Added `core/data_io.py` (was missing entirely)
- Added `hop-sim/visual.py` and `hop_sim_demo.py` (existed but were omitted)
- Replaced `transient/  # to be built` with the full 7-file listing
- Added a **Transient** section to the "How to Run" block

---

## core/data_io.py — multi-kh support

`save_sweep` and `load_sweep` were extended to handle sweeps over multiple hop rates in a single file.

**`save_sweep`**
- `total_photons` and `total_photons_homo` now accept either a plain `ndarray` (old behaviour, unchanged) or a `dict {k_h: ndarray}` (new multi-kh mode).
- Multi-kh writes columns named `total_photons_kh1`, `total_photons_kh10`, etc., and stores `# k_h_values = 1,10,100` in the header.
- Single-series files are written identically to before — no breaking change.

**`load_sweep`** return signatures:
| File format | Returns |
|---|---|
| Single-series (1p / 2p) | `(kex, total_photons, params)` |
| Single-series TTA with homo | `(kex, total_photons, total_photons_homo, params)` |
| Multi-kh, no homo | `(kex, photons_dict, params)` |
| Multi-kh with homo | `(kex, photons_dict, photons_homo_dict, params)` |

Dicts are keyed by `float` k_h values parsed from the `k_h_values` header field.

---

## transient/sweep_tta.py — multi-kh sweep

- `KH_TTA = 10.0` replaced with `KH_LIST = [...]` — set any list of hop rates.
- Outer loop over `KH_LIST`, inner loop over `kex_range`; results stored as dicts.
- `k_h` removed from the saved `params` dict (it is now encoded in the column names and `k_h_values` header).
- Per-run timing added (`time.time()`), printed via `tqdm.write` so it doesn't clobber the progress bar.

---

## transient/plot_tta.py — full rewrite

**Control panel** (top of file):
```python
KH_PLOT   = None   # None → all k_h in file; or e.g. [1.0, 100.0]
SHOW_HOMO = True   # dashed homo lines
SHOW_SS   = True   # steady-state reference lines
```

**Style conventions established here (carried into comparisons/):**
- Color encodes model / k_h; **solid = hetero, dashed = homo**; no markers; thin lines (`lw=0.9`).
- SS lines: blue (`#1565C0`); transient lines: non-blue palette cycling per k_h.

**Other changes:**
- `_normalize_load` helper handles both old single-kh and new multi-kh files transparently.
- Params printed to console on load.
- Title omits `t_off` and `n_pulses` when `mode != "pulse_train"`.
- SS sweeps run inside the k_h loop, one per k_h.

---

## transient/transient_pulse.py — run control + timing

**Run control panel** added at the top of the parameters section:
```python
RUN_MODELS = ["1p", "2p", "tta"]   # any subset
```
Each model's simulation and plot block is now gated by `if "model" in RUN_MODELS`.

**Timing:** `import time` added; each run prints its wall time appended to the existing dt/steps line:
```
Running TTA ...
  dt = 1.00e-02 ns  (20000 steps)  2.34s
```

**Title** now lists `models=RUN_MODELS` and only includes TTA params when `"tta"` is in the list.

---

## comparisons/plot_comparison.py — new file

Unified comparison plot that can load and overlay 1p, 2p, and TTA data in a single figure.

**File selectors** — independent per model:
```python
DATE_1P = None;  N_1P = 0
DATE_2P = None;  N_2P = 0
DATE_TTA = None; N_TTA = 0
```

**Plot control:**
```python
SHOW_1P / SHOW_2P / SHOW_TTA         # master on/off per model
SHOW_1P_TRANSIENT / SHOW_2P_TRANSIENT  # False → CW/SS only, no file needed
SHOW_SS                               # CW reference lines for all active models
SHOW_HOMO                             # TTA dashed homo lines
KH_PLOT                               # subset of TTA k_h values to plot
```

**Standalone SS params** — used when `SHOW_Xp_TRANSIENT = False` (no file loaded) or as fallback if a file is not found:
```python
KEX_SS_MIN / KEX_SS_MAX / KEX_SS_N
K_FLUOR_1P
K_1_2P / K_FLUOR_2P / G_FACTOR_2P    # G_FACTOR_2P = None → sqrt(T_total/t_on) from file
```

**Color / style conventions:**
- Each model has a `(transient_color, SS_color)` pair — same hue, light shade for pulsed, dark shade for CW.
- `COLOR_1P = ("#90CAF9", "#1565C0")` — light/dark blue
- `COLOR_2P = ("#EF9A9A", "#B71C1C")` — light/dark red
- `TTA_COLORS` — list of pairs cycling green → brown → purple → teal → olive
- Solid = hetero, dashed = homo; no markers; `LW = 0.9`

**Legend labels:** `"1p pulsed"`, `"1p CW"`, `"TTA k_h=1 pulsed"`, `"TTA k_h=1 CW homoTTA"`, etc.

**Title** (two lines, second line only when a transient file was loaded):
```
Model Comparison  |  Pulsed (light)  ·  CW (dark)  ·  homoTTA (dashed)
pulse width = 100 ns  ·  Rep Rate = 5e+06 Hz
```

Pulse metrics are collected from the first successfully loaded transient file (`pulse_params` collector).

**On load**, each file prints its path, k_ex range, available k_h values, and all non-bookkeeping params to the console.

---

## comparisons/plot_comparison.py — multi-TTA-file support + color overhaul (2026-04-30)

### Multi-file TTA loading

`DATE_TTA / N_TTA` replaced with a `TTA_FILES` list of dicts:
```python
TTA_FILES = [
    {"date": None, "n": 0},          # most recent file today
    {"date": None, "n": 1},          # second most recent today
    # {"date": "2026-04-28", "n": 0},   # specific date
    # {"date": None, "n": 1, "kh_plot": [1.0]},  # per-file k_h override
]
```
Each entry is a separate sweep file (different pulse parameters, different `t_on`, etc.).  
A global `tta_color_idx` increments across all files × k_h values so colors never repeat between files.  
Per-file `kh_plot` key overrides the global `KH_PLOT` for that file only.

### Color palette overhaul

Adopted the proto7 tab: color palette. Dropped the light/dark shade convention; instead, pulsed and CW lines use **completely separate named color variables**:

```python
COLOR_1P_SS    = "blue";          COLOR_1P    = "cornflowerblue"
COLOR_2P_SS    = "red";           COLOR_2P    = "tomato"
TTA_COLORS     = ["tab:orange", "tab:green", "tab:purple", "tab:brown", ...]
TTA_SS_COLORS  = ["goldenrod",  "darkgreen",  "orchid",    "sienna",   ...]
```

### Label and title updates

- Pulsed TTA labels include `t_on`: `"TTA  k_h=10  t_on=100  pulsed"`
- CW TTA labels: `"TTA  k_h=10  CW"`
- Title: `"Model Comparison  |  Pulsed / CW (separate colors)  ·  homoTTA (dashed)"`
- Second title line (when any transient file loaded): `"pulse width = 100 ns  ·  Rep Rate = 5e+06 Hz"`
