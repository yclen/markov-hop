import argparse
import multiprocessing as mp

from visual import run_simulation
from monte_carlo import run_monte_carlo, plot_monte_carlo


def _run_visual(kh, n_mediators, step_size, n_sims, bin_mult, units):
    mc_processes = []

    def on_p():
        p = mp.Process(target=_run_monte_carlo, args=(kh, n_mediators, step_size, n_sims, bin_mult, units))
        p.start()
        mc_processes.append(p)

    def on_q():
        for p in mc_processes:
            if p.is_alive():
                p.terminate()

    run_simulation(kh=kh, n_mediators=n_mediators, on_p=on_p, on_q=on_q)


def _run_monte_carlo(kh, n_mediators, step_size, n_sims, bin_mult, units):
    results_ns, mean_ns, median_ns, d = run_monte_carlo(
        hop_rate=kh,
        step_size=step_size,
        n_mediators=n_mediators,
        n_sims=n_sims,
    )
    plot_monte_carlo(
        results_ns=results_ns,
        mean_ns=mean_ns,
        step_size=step_size,
        n_mediators=n_mediators,
        hop_rate=kh,
        n_sims=n_sims,
        bin_mult=bin_mult,
        units=units,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",         type=int,   default=6,       help="number of mediators")
    parser.add_argument("--kh",        type=float, default=5.0,    help="hopping rate")
    parser.add_argument("--step-size", type=float, default=0.01,    help="time per step (ns)")
    parser.add_argument("--sims",      type=int,   default=100_000, help="number of simulations")
    parser.add_argument("--bin",       type=float, default=60,       help="bin width multiplier")
    parser.add_argument("--u",         type=str,   default="s",     help="time units for plot")
    args = parser.parse_args()

    p_visual = mp.Process(target=_run_visual, args=(args.kh, args.m, args.step_size, args.sims, args.bin, args.u))

    p_visual.start()
    p_visual.join()
