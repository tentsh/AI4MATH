"""
Arjevani zero-chain: stochastic gradient oracle and single Proj-SGD trajectory.
Builds on arjevani_full.py.
"""

import numpy as np
from arjevani_rotated import make_U
from arjevani_full import F_scaled, grad_F_scaled, make_lambda, sample_ball

R = 0.5   # ball radius / projection radius


# ---------------------------------------------------------------------------
# Oracle and projection
# ---------------------------------------------------------------------------

def project(x: np.ndarray) -> np.ndarray:
    """Project onto closed ball of radius R = 0.5."""
    norm = np.linalg.norm(x)
    if norm == 0.0:
        return x.copy()
    return x * min(1.0, R / norm)


def stoch_grad(x: np.ndarray, U: np.ndarray, lam: float, L_target: float,
               batch_size: int, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """grad_F_scaled(x) + N(0, sigma^2 / batch_size * I)."""
    noise = rng.standard_normal(len(x)) * sigma / np.sqrt(batch_size)
    return grad_F_scaled(x, U, lam, L_target) + noise


def stat_estimate(x: np.ndarray, U: np.ndarray, lam: float, L_target: float,
                  sigma: float, rng: np.random.Generator,
                  eval_batch_size: int = 200) -> float:
    """Stochastic projected-gradient stationarity measure."""
    g     = stoch_grad(x, U, lam, L_target, eval_batch_size, sigma, rng)
    x_new = project(x - (1.0 / L_target) * g)
    return float(np.linalg.norm(x_new - x) * L_target)


# ---------------------------------------------------------------------------
# Single Proj-SGD run
# ---------------------------------------------------------------------------

def run_sgd(T=10, d=20, epsilon=0.2, L_target=1.0, sigma=1.0,
            N_max=200_000, eval_every=500, print_every=1000,
            init_seed=0, sgd_seed=42):

    U   = make_U(d, T, seed=0)
    lam = make_lambda(epsilon, L_target)

    rng_init = np.random.default_rng(init_seed)
    x = sample_ball(1, d, R, rng_init)[0]
    rng = np.random.default_rng(sgd_seed)

    print(f"T={T}, d={d}, epsilon={epsilon}, sigma={sigma}")
    print(f"lambda={lam:.4f}   L_target={L_target}   N_max={N_max:,}")
    print(f"Schedule B: eta_t = min(1/L, 1/sqrt(t)),  batch_size=1")
    print(f"Initial ||x|| = {np.linalg.norm(x):.6f}\n")
    print(f"{'Step':>8}  {'Samples':>9}  {'stat_est':>10}  {'||x||':>8}")
    print("-" * 46)

    t              = 0
    total_samples  = 0
    path_length    = 0.0
    converged      = False
    last_stat      = float('inf')

    # Evaluate stationarity at t=0 (counts toward sample budget)
    last_stat      = stat_estimate(x, U, lam, L_target, sigma, rng)
    total_samples += 200

    while total_samples < N_max:
        t += 1

        # SGD step (schedule B, batch_size=1)
        eta   = min(1.0 / L_target, 1.0 / np.sqrt(t))
        g     = stoch_grad(x, U, lam, L_target, 1, sigma, rng)
        total_samples += 1

        x_new       = project(x - eta * g)
        path_length += np.linalg.norm(x_new - x)
        x           = x_new

        # Stationarity check every eval_every steps
        if t % eval_every == 0:
            last_stat      = stat_estimate(x, U, lam, L_target, sigma, rng)
            total_samples += 200
            if last_stat <= epsilon:
                converged = True
                if t % print_every != 0:         # print final state if mid-interval
                    print(f"{t:>8}  {total_samples:>9}  {last_stat:>10.4f}  "
                          f"{np.linalg.norm(x):>8.4f}")
                break

        # Progress print every print_every steps
        if t % print_every == 0:
            print(f"{t:>8}  {total_samples:>9}  {last_stat:>10.4f}  "
                  f"{np.linalg.norm(x):>8.4f}")

    print(f"\n{'='*46}")
    print(f"Total samples N  : {total_samples:,}")
    print(f"Steps            : {t:,}")
    print(f"Final stat_est   : {last_stat:.6f}")
    print(f"Success          : {converged}")
    print(f"Path length      : {path_length:.4f}")
    return total_samples, last_stat, converged, path_length


if __name__ == '__main__':
    run_sgd()
