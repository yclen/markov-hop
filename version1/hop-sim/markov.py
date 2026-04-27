import numpy as np
import matplotlib.pyplot as plt
import time


def build_matrix(Ph, n_mediators):
    N = n_mediators + 2
    M = np.zeros((N, N))

    # State 0: reflective, only hops right
    M[0, 0] = 1 - Ph
    M[1, 0] = Ph

    # State 1: special — backward hop is clamped, stays at 1
    M[1, 1] = 1 - Ph/2
    M[2, 1] = Ph/2

    # States 2 to N-2: true bidirectional hopping
    for i in range(2, N - 1):
        M[i-1, i] = Ph/2
        M[i,   i] = 1 - Ph
        M[i+1, i] = Ph/2

    # State N-1: absorbing
    M[N-1, N-1] = 1.0

    return M


def run_markov(kh=10.0, n_steps=100, n_mediators=1):
    """
    Simulate a 1D random walk with matrix multiplication.

    Parameters
    ----------
    kh          : float - hopping rate (ns^-1)
    n_steps     : int   - number of matrix multiplication steps
    n_mediators : int   - number of mediator states

    Returns
    -------
    history : np.ndarray, shape (n_steps+1, N) - probability distribution over time
    times   : np.ndarray, shape (n_steps+1,)   - time axis in ns
    mean_t  : float                             - mean absorption time in ns
    """

    Ph = 0.1
    dt = Ph / kh
    

    M = build_matrix(Ph, n_mediators)

    N = n_mediators + 2
    v = np.zeros(N)
    v[0] = 1.0

    history = [v.copy()]
    t_start = time.time()
    for _ in range(n_steps):
        v = M @ v
        history.append(v.copy())
    t_end = time.time()

    history = np.array(history)
    times   = np.arange(n_steps + 1) * dt

    cdf    = history[:, -1]
    pdf    = np.diff(cdf, prepend=0) / dt
    mean_t = np.sum((1 - cdf) * dt)
    duration = t_end - t_start

    print(f"Steps completed        : {n_steps}")
    print(f"Time elapsed           : {round(duration,6)} s")
    print(f"Absorbed               : {cdf[-1]:.4f}")
    print(f"Mean absorption time   : {mean_t:.4f} ns")

    return history, times, mean_t, duration



def get_migration_time(kh=10.0, n_mediators=1):
    history, times, mean_t, duration = run_markov(kh=kh, n_steps=80*n_mediators**2, n_mediators=n_mediators)
    return mean_t

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",     type=int,   default=5,    help="number of mediators")
    parser.add_argument("--kh",    type=float, default=10.0,  help="hopping rate (ns^-1)")
    parser.add_argument("--steps", type=int,   default=2000, help="number of steps")
    args = parser.parse_args()

    KH          = args.kh
    N_MEDIATORS = args.m
    N_STEPS     = 80*N_MEDIATORS**2

    history, times, mean_t, duration = run_markov(kh=KH, n_steps=N_STEPS, n_mediators=N_MEDIATORS)

    cdf = history[:, -1]
    pdf = np.diff(cdf, prepend=0) / (times[1] - times[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: PDF
    ax = axes[0]
    ax.plot(times, pdf, color="#4a90d9", linewidth=2)
    ax.axvline(mean_t, color="red", linewidth=1.5, linestyle=":", label=f"Mean: {mean_t:.3f} ns")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"PDF  |  kh={KH}/ns  |  N_mediators={N_MEDIATORS}")
    ax.legend()

    # Right: CDF
    ax = axes[1]
    ax.plot(times, cdf, color="#f5a623", linewidth=2)
    ax.axvline(mean_t, color="red", linewidth=1.5, linestyle=":", label=f"Mean: {mean_t:.3f} ns")
    ax.axhline(0.5,    color="gray", linewidth=1,   linestyle="--", label="50% absorbed")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF  |  kh={KH}/ns  |  N_mediators={N_MEDIATORS}")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.show()