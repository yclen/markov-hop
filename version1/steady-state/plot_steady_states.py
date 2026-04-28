import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from ss_1p  import sweep_1p
from ss_2p  import sweep_2p
from ss_tta import sweep_tta

# ── Shared ────────────────────────────────────────────────────────────────────

KEX_RANGE = np.logspace(-8, 2, 200)

# ── 1p params ─────────────────────────────────────────────────────────────────

K_FLUOR_1P = 0.1

# ── 2p params ─────────────────────────────────────────────────────────────────

K_1        = 1e5      # S1 decay rate
K_FLUOR_2P = 0.1
G_FACTOR   = np.sqrt(1e5)      # CW; set to G_FACTOR from models for pulsed
# k_ex2 tracks k_ex (k_ex2=None)

# ── TTA params (main model) ───────────────────────────────────────────────────

N_MED      = 4
KH_VALUES  = [1.0, 10.0, 100.0]
K_DECAY    = 0.001
F_SPIN     = 0.4

# ── Run sweeps ────────────────────────────────────────────────────────────────

print("Running 1p sweep ...")
kex_1p, k_emit_1p = sweep_1p(KEX_RANGE, k_fluor=K_FLUOR_1P)

print("Running 2p sweep ...")
kex_2p, k_emit_2p = sweep_2p(KEX_RANGE, k_1=K_1, k_fluor=K_FLUOR_2P,
                               k_ex2=None, g_factor=G_FACTOR)

tta_results = []
for k_h in KH_VALUES:
    print(f"Running TTA sweep (N={N_MED}, kh={k_h}) ...")
    kex, k_emit, k_homo = sweep_tta(KEX_RANGE, n_med=N_MED, k_h=k_h,
                                     k_decay=K_DECAY, f_spin=F_SPIN)
    tta_results.append((k_h, kex, k_emit))

# ── Log-derivative helper ─────────────────────────────────────────────────────

def log_deriv(kex, k_emit):
    log_x = np.log(kex)
    log_y = np.log(np.clip(k_emit, 1e-300, None))
    return np.gradient(log_y, log_x)

# ── Plot ──────────────────────────────────────────────────────────────────────

TTA_COLORS = ["#E87722", "#9B59B6", "#2ECC71", "#E74C3C", "#1ABC9C"]

panels = [
    ("linear", "linear", "Linear"),
    ("log",    "log",    "Log-log"),
    ("log",    "linear", "Log-derivative  d(log k_emit) / d(log k_ex)"),
]

for xscale, yscale, title in panels:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    is_deriv  = title.startswith("Log-deriv")

    def y(kex, k_emit):
        return log_deriv(kex, k_emit) if is_deriv else k_emit

    # 1p
    ax.plot(kex_1p, y(kex_1p, k_emit_1p),
            color="#2E86C1", linewidth=1.2, linestyle="-", label=f"1p  kfluor={K_FLUOR_1P}")

    # 2p
    ax.plot(kex_2p, y(kex_2p, k_emit_2p),
            color="#C0392B", linewidth=1.2, linestyle="-", label=f"2p  k1={K_1:.0e}  kfluor={K_FLUOR_2P}")

    # TTA
    for i, (k_h, kex, k_emit) in enumerate(tta_results):
        ax.plot(kex, y(kex, k_emit),
                color=TTA_COLORS[i % len(TTA_COLORS)], linewidth=1.2, linestyle="-",
                label=f"TTA  N={N_MED}  kh={k_h}")

    if is_deriv:
        ax.axhline(1.0, color="lightgray", linestyle="--", linewidth=0.8)
        ax.axhline(2.0, color="lightgray", linestyle=":",  linewidth=0.8)
        ax.set_ylabel("d log k_emit / d log k_ex")
    else:
        ax.set_yscale(yscale)
        ax.set_ylabel("k_emit  (ns⁻¹)")

    ax.set_xscale(xscale)
    ax.set_xlabel("k_ex  (ns⁻¹)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Steady-state emission  |  N={N_MED}  kdecay={K_DECAY}  f_spin={F_SPIN}")

plt.show()
