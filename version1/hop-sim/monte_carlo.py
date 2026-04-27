import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_monte_carlo(hop_rate=10.0, step_size=0.01, n_mediators=1, n_sims=1_000_000):
    """
    Stochastic hopping simulation across a 1D chain.

    Parameters
    ----------
    hop_rate    : float - hopping rate (ns^-1)
    step_size   : float - time per step (ns)
    n_mediators : int   - number of mediator states between sensitizer and emitter
    n_sims      : int   - number of simulation runs

    Returns
    -------
    results_ns  : np.ndarray - absorption times in ns for each run
    mean_ns     : float      - mean absorption time (ns)
    median_ns   : float      - median absorption time (ns)
    """
    hop_prob    = hop_rate * step_size
    n_molecules = n_mediators + 2
    emitter_idx = n_molecules - 1

    assert 0 <= hop_prob <= 1, "hop_rate * step_size must be <= 1"

    def run_sim():
        idx, steps = 0, 0
        while idx != emitter_idx:
            if random.random() < hop_prob:
                if random.random() < 0.5:
                    idx = min(idx + 1, emitter_idx)
                else:
                    idx = max(idx - 1, 1)
            steps += 1
        return steps

    t_start       = time.time()
    results_steps = [run_sim() for _ in tqdm(range(n_sims), desc="Simulating")]
    t_end         = time.time()

    results_ns = np.array(results_steps) * step_size
    mean_ns    = np.mean(results_ns)
    median_ns  = np.median(results_ns)
    duration = t_end - t_start

    print(f"\nSimulations : {n_sims:,}")
    print(f"Mediators   : {n_mediators}")
    print(f"Hop rate    : {hop_rate} per ns  (P={hop_prob} per step)")
    print(f"Step size   : {step_size} ns")
    print(f"Sim duration: {t_end - t_start:.3f} s")
    print(f"Mean time   : {mean_ns:.4f} ns")
    print(f"Median time : {median_ns:.4f} ns")

    return results_ns, mean_ns, median_ns, duration


def plot_monte_carlo(results_ns, mean_ns, step_size=0.01, n_mediators=1, hop_rate=10.0,
                     n_sims=100_000, bin_mult=1, units="ns"):
    bin_width = step_size * bin_mult
    bins = np.arange(0, results_ns.max() + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(results_ns, bins=bins, color="#4a90d9", edgecolor="none", alpha=0.85)
    ax.axvline(mean_ns, color="#f5a623", linewidth=2, label=f"Mean: {mean_ns:.3f} {units}")
    ax.set_xlabel(f"Time to reach emitter ({units})", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Hopping simulation  |  N_mediators={n_mediators}  |  k_hop={hop_rate}/{units}  |  t_hop={1/hop_rate:.4g} {units}  |  N={n_sims:,}",
        fontsize=13
    )
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",         type=int,   default=5,       help="number of mediators")
    parser.add_argument("--kh",        type=float, default=10.0,    help="hopping rate (ns^-1)")
    parser.add_argument("--step-size", type=float, default=0.01,    help="time per step (ns)")
    parser.add_argument("--sims",      type=int,   default=100_000, help="number of simulations")
    parser.add_argument("--bin",       type=float, default=6,       help="bin width multiplier (1 = step size)")
    parser.add_argument("--u",         type=str,   default="ns",    help="time units for plot")
    args = parser.parse_args()

    results_ns, mean_ns, median_ns, d = run_monte_carlo(
        hop_rate=args.kh,
        step_size=0.1/args.kh,
        n_mediators=args.m,
        n_sims=args.sims,
    )

    plot_monte_carlo(
        results_ns=results_ns,
        mean_ns=mean_ns,
        step_size=args.step_size,
        n_mediators=args.m,
        hop_rate=args.kh,
        n_sims=args.sims,
        bin_mult=args.bin,
        units=args.u,
    )