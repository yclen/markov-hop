# markov-hop / version1

## Overview

This project models **Triplet-Triplet Annihilation Upconversion (TTA-UC)** using Markov chain (rate matrix) methods. A chain of molecular sites is simulated:

```
Sensitizer → M1 → M2 → ... → MN → Annihilator
```

Triplet excitons hop along the chain and undergo annihilation at the annihilator site. The goal is to understand how chain length (N), hopping rate (k_h), and excitation rate (k_ex) govern emission behaviour — in particular the quadratic-to-linear transition in TTA emission.

The TTA model is compared against a **1-photon** (linear) fluorophore and a **2-photon** sequential absorption model to contextualise its power dependence.

---

## Key Parameters

| Parameter | Symbol | Typical units |
|-----------|--------|--------------|
| Hopping rate | `k_h` | ns⁻¹ |
| Excitation rate | `k_ex` | ns⁻¹ |
| Decay rate | `k_decay` | ns⁻¹ |
| Number of mediators | `N` | integer |
| Spin-statistical factor | `f_spin` | 0–1 (default 0.4) |
| Pulsed enhancement | `g_factor` | dimensionless |

---

## Folder Structure

```
version1/
├── core/                   # Shared computation modules
│   ├── models.py           # TTA, 1p, 2p Markov matrix builders and rate solvers
│   ├── data_io.py          # CSV sweep I/O helpers (save_sweep, get_file, load_sweep)
│   └── view_matrix.py      # CLI tool to visualise any model's transition matrix
│
├── hop-sim/                # Basic 1D random walk — migration time analysis
│   ├── markov.py           # Transition matrix and mean absorption time
│   ├── monte_carlo.py      # Stochastic hopping simulation
│   ├── compare.py          # Markov vs Monte Carlo comparison plot
│   ├── param_sweep_hop.py  # Sweep k_h and N, saves migration_times.csv
│   ├── plot_migration_times.py  # 3D surface + power-law fits
│   ├── visual.py           # Interactive pygame visualisation of hopping triplet
│   └── hop_sim_demo.py     # Launches visual window with Monte Carlo on keypress
│
├── steady-state/           # Steady-state emission rate vs k_ex sweeps
│   ├── ss_1p.py            # sweep_1p() — 1-photon model
│   ├── ss_2p.py            # sweep_2p() — 2-photon sequential model
│   ├── ss_tta.py           # sweep_tta() — TTA chain model
│   └── plot_steady_states.py  # Runs all three sweeps and plots 3-panel comparison
│
├── transient/              # Time-domain (pulsed excitation) simulations
│   ├── transient_pulse.py  # Core engine: make_excitation_profile, run_1p/2p/tta
│   ├── sweep_1p.py         # k_ex sweep for 1-photon transient, saves CSV
│   ├── sweep_2p.py         # k_ex sweep for 2-photon transient, saves CSV
│   ├── sweep_tta.py        # k_ex sweep for TTA transient (homo + hetero), saves CSV
│   ├── plot_1p.py          # Transient vs steady-state overlay for 1-photon
│   ├── plot_2p.py          # Transient vs steady-state overlay for 2-photon
│   └── plot_tta.py         # Transient vs steady-state overlay for TTA (4 curves)
│
├── comparisons/            # Cross-model analysis (to be built)
│
└── data/                   # Generated data — not tracked by git
    └── YYYY-MM-DD/         # Date-stamped subfolders for run outputs
```

---

## How to Run

**View a transition matrix:**
```bash
cd core
python view_matrix.py --model tta --n 2 --kh 10 --kex 1 --kdecay 0.01
python view_matrix.py --model 1p --kex 1 --kfluor 0.1
python view_matrix.py --model 2p --kex 1 --k1 1e8
```

**Steady-state sweep and plot:**
```bash
cd steady-state
python plot_steady_states.py
```

**Migration time sweep:**
```bash
cd hop-sim
python param_sweep_hop.py     # generates migration_times.csv
python plot_migration_times.py
```

**Transient (pulsed) sweep and plot:**
```bash
cd transient
python sweep_tta.py           # generates TTA transient CSV (hetero + homo columns)
python plot_tta.py            # transient vs steady-state overlay
python sweep_1p.py && python plot_1p.py
python sweep_2p.py && python plot_2p.py
```
