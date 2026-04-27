import numpy as np
import matplotlib.pyplot as plt
from markov import run_markov
from monte_carlo import run_monte_carlo


def run_compare(kh=10.0, n_mediators=5, step_size=0.01, n_sims=200_000, bin_mult=1, units="ns"):
    results_ns, mean_stoch, median, monte_duration = run_monte_carlo(
        hop_rate=kh,
        step_size=step_size,
        n_mediators=n_mediators,
        n_sims=n_sims,
    )

    n_steps = int(results_ns.max() / step_size)

    history, times, mean_markov, markov_duration = run_markov(kh=kh, n_steps=n_steps, n_mediators=n_mediators)
    cdf = history[:, -1]
    pdf = np.diff(cdf, prepend=0) / (times[1] - times[0])

    print(f"\nMarkov duration     : {markov_duration:.6f} s")
    print(f"Monte Carlo duration: {monte_duration:.3f} s")
    print(f"Speedup             : {monte_duration / markov_duration:.0f}x")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bins = np.arange(0, results_ns.max() + step_size, step_size * bin_mult)
    ax1.hist(results_ns, bins=bins, density=True, color="#4a90d9", edgecolor="none", alpha=0.6,
             label=f"Monte Carlo (n={n_sims:,})  |  Runtime: {monte_duration:.2f} s")

    ax1.plot(times, pdf, color="red", linewidth=2,
             label=f"Markov (analytical)  |  Runtime: {markov_duration:.4f} s")

    ax1.axvline(mean_stoch,  color="#4a90d9", linewidth=1.5, linestyle=":",
                label=f"Mean Monte Carlo : {mean_stoch:.3f} {units}")
    ax1.axvline(mean_markov, color="red",     linewidth=1.5, linestyle=":",
                label=f"Mean Markov      : {mean_markov:.3f} {units}")

    ax1.set_xlabel(f"Migration time ({units})", fontsize=13)
    ax1.set_ylabel("Probability density", fontsize=13)
    ax1.set_ylim(0)

    ax2.set_ylim(0, ax1.get_ylim()[1] * n_sims * step_size * bin_mult)
    ax2.set_ylabel("Counts", fontsize=13)
    ax2.tick_params(axis="y", colors="black")

    ax1.set_title(
        f"Markov vs Monte Carlo  |  t_hop={1/kh:.4g} {units}  |  k_hop={kh}/{units}  |  N_mediators={n_mediators}",
        fontsize=13
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)

    plt.tight_layout()
    plt.savefig("compare.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",         type=int,   default=5,       help="number of mediators")
    parser.add_argument("--kh",        type=float, default=10.0,    help="hopping rate")
    parser.add_argument("--step-size", type=float, default=0.01,    help="time per step (ns)")
    parser.add_argument("--sims",      type=int,   default=200_000, help="number of simulations")
    parser.add_argument("--bin",       type=float, default=1,       help="bin width multiplier")
    parser.add_argument("--u",         type=str,   default="ns",    help="time units for plot")
    args = parser.parse_args()

    run_compare(
        kh=args.kh,
        n_mediators=args.m,
        step_size=0.1/args.kh,
        n_sims=args.sims,
        bin_mult=args.bin,
        units=args.u,
    )
