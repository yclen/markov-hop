import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models import (
    build_tta_matrix, build_1p_matrix, build_2p_matrix,
    P_MAX, G_FACTOR,
)

ANNOTATE_THRESHOLD = 32   # hide per-cell text above this many states


def _plot_matrix(M, labels, title):
    N = M.shape[0]
    annotate = N <= ANNOTATE_THRESHOLD

    fig, ax = plt.subplots(figsize=(max(5, N * 0.55), max(4, N * 0.5)))
    im = ax.imshow(M.T, cmap="Blues", vmin=0, vmax=1)

    if annotate:
        for i in range(N):
            for j in range(N):
                val = M[j, i]
                if val > 0:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="black" if val < 0.6 else "white")

    ax.set_xlabel("To state", fontsize=11)
    ax.set_ylabel("From state", fontsize=11)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="Transition probability")
    plt.tight_layout()
    plt.show()


def _tta_labels(n_med):
    """Bit-string state labels: e.g. 'S·M1·M2·A' → '1·0·1·0'."""
    n_sites = n_med + 2
    site_names = ["S"] + [f"M{i}" for i in range(1, n_med + 1)] + ["A"]
    labels = []
    for s in range(2 ** n_sites):
        bits = [(s >> (n_sites - 1 - i)) & 1 for i in range(n_sites)]
        labels.append("".join(str(b) for b in bits))
    return labels, site_names


# ── TTA ───────────────────────────────────────────────────────────────────────

def view_tta(n_med, k_h, k_ex, k_decay, homo_tta, f_spin):
    dt     = P_MAX / max(k_h, k_ex, k_decay)
    Ph     = k_h     * dt
    Pex    = k_ex    * dt
    Pdecay = k_decay * dt

    M, emit_mask, homo_mask = build_tta_matrix(
        n_med=n_med, Ph=Ph, Pex=Pex, Pdecay=Pdecay, homo_tta=homo_tta, f_spin=f_spin,
    )

    labels, site_names = _tta_labels(n_med)
    col_sums = M.sum(axis=0)

    print(f"\nTTA matrix  |  N={n_med}  kh={k_h}  kex={k_ex}  kdecay={k_decay}  f_spin={f_spin}"
          + ("  homoTTA" if homo_tta else ""))
    print(f"Sites: {' | '.join(site_names)}")
    print(f"States: {2 ** (n_med + 2)}  |  Matrix: {M.shape[0]}x{M.shape[1]}")
    print(f"Column sums OK: {np.allclose(col_sums, 1.0)}")
    print(f"HeteroTTA states: {[labels[s] for s in range(len(labels)) if emit_mask[s] > 0]}")
    if homo_tta:
        print(f"HomoTTA states:   {[labels[s] for s in range(len(labels)) if homo_mask[s] > 0]}")

    title = (f"TTA transition matrix  |  N={n_med}  kh={k_h}  kex={k_ex}  kdecay={k_decay}"
             + ("  homoTTA" if homo_tta else ""))
    _plot_matrix(M, labels, title)


# ── 1-Photon ──────────────────────────────────────────────────────────────────

def view_1p(k_ex, k_fluor):
    dt     = P_MAX / max(k_ex, k_fluor)
    Pex    = k_ex    * dt
    Pfluor = k_fluor * dt

    M, emit_mask = build_1p_matrix(Pex, Pfluor)
    labels = ["Ground", "Excited"]

    print(f"\n1-Photon matrix  |  kex={k_ex}  kfluor={k_fluor}")
    print(f"Column sums OK: {np.allclose(M.sum(axis=0), 1.0)}")
    print(f"Emission from: {[labels[i] for i, e in enumerate(emit_mask) if e > 0]}")
    header = f"{'':12}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "".join(f"{M[i,j]:>12.6f}" for j in range(len(labels)))
        print(f"{row_label:12}{row}")

    _plot_matrix(M, labels, f"1-Photon transition matrix  |  kex={k_ex}  kfluor={k_fluor}")


# ── 2-Photon ──────────────────────────────────────────────────────────────────

def view_2p(k_ex, k_ex2, k_1, k_fluor, g_factor):
    dt     = P_MAX / max(k_ex, k_ex2, k_1, k_fluor)
    Pex    = k_ex    * dt
    Pex2   = k_ex2   * dt
    P1     = k_1     * dt
    Pfluor = k_fluor * dt

    M, emit_mask = build_2p_matrix(Pex, Pex2, P1, Pfluor)
    labels = ["Ground", "Intermediate", "Excited"]

    print(f"\n2-Photon matrix  |  kex={k_ex}  kex2={k_ex2}  k1={k_1}  kfluor={k_fluor}  g={g_factor:.0f}")
    print(f"Column sums OK: {np.allclose(M.sum(axis=0), 1.0)}")
    print(f"Emission from: {[labels[i] for i, e in enumerate(emit_mask) if e > 0]}")
    header = f"{'':16}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "".join(f"{M[i,j]:>14.6f}" for j in range(len(labels)))
        print(f"{row_label:16}{row}")

    _plot_matrix(M, labels,
                 f"2-Photon transition matrix  |  kex={k_ex}  kex2={k_ex2}  k1={k_1}  kfluor={k_fluor}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise a Markov transition matrix.")
    parser.add_argument("--model",    choices=["tta", "1p", "2p"], required=True)

    # TTA args
    parser.add_argument("--n",        type=int,   default=2,    help="[TTA] number of mediators")
    parser.add_argument("--kh",       type=float, default=10.0, help="[TTA] hopping rate")
    parser.add_argument("--kdecay",   type=float, default=0.01, help="[TTA] decay rate")
    parser.add_argument("--homo-tta", action="store_true",      help="[TTA] enable homoTTA")
    parser.add_argument("--f-spin",   type=float, default=0.4,  help="[TTA] spin-statistical factor (default 0.4)")

    # shared / 1p / 2p args
    parser.add_argument("--kex",      type=float, default=1.0,  help="excitation rate")
    parser.add_argument("--kfluor",   type=float, default=0.1,  help="[1p/2p] fluorescence rate")
    parser.add_argument("--kex2",     type=float, default=1.0,  help="[2p] S1→upper excitation rate")
    parser.add_argument("--k1",       type=float, default=1e5, help="[2p] S1 decay rate")
    parser.add_argument("--g-factor", type=float, default=1.0,
                                                               help="[2p] pulsed enhancement (default 1.0 = CW)")
    args = parser.parse_args()

    if args.model == "tta":
        view_tta(n_med=args.n, k_h=args.kh, k_ex=args.kex,
                 k_decay=args.kdecay, homo_tta=args.homo_tta, f_spin=args.f_spin)
    elif args.model == "1p":
        view_1p(k_ex=args.kex, k_fluor=args.kfluor)
    elif args.model == "2p":
        view_2p(k_ex=args.kex, k_ex2=args.kex2, k_1=args.k1,
                k_fluor=args.kfluor, g_factor=args.g_factor)
