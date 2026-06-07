"""
T-sweep experiment: Arjevani zero-chain with T varied independently of epsilon.
Reuses low-level building blocks from arjevani_experiment.py.
"""

import numpy as np
import csv
import os
import time
from itertools import product

from arjevani_rotated import make_U
from arjevani_full import make_lambda, sample_ball
from arjevani_sgd import project, stoch_grad, stat_estimate

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
L_TARGET   = 1.0
SIGMA      = 1.0
T_VALUES   = [10, 25, 50]
EPS_GRID   = [0.2, 0.15, 0.1]
N_SEEDS    = 3
N_MAX      = 1_000_000
EVAL_EVERY = 500
R          = 0.5
SCHEDULE   = 'B'
TRAJ_DIR   = 'results/trajectories'
RESULTS_FILE = 'results/results_t_sweep.csv'

TOTAL_RUNS = len(T_VALUES) * len(EPS_GRID) * N_SEEDS   # 36


# ---------------------------------------------------------------------------
# Single run (T passed explicitly)
# ---------------------------------------------------------------------------

def _save_trajectory(T: int, epsilon: float, seed: int, traj: list) -> None:
    os.makedirs(TRAJ_DIR, exist_ok=True)
    fname = os.path.join(TRAJ_DIR, f"trajectory_B_T{T}_{epsilon}_{seed}.csv")
    with open(fname, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step', 'last_coord'])
        w.writeheader()
        w.writerows({'step': s, 'last_coord': c} for s, c in traj)


def run_single_T(T: int, epsilon: float, seed: int) -> dict:
    d   = 2 * T
    U   = make_U(d, T, seed=0)          # U fixed by (T, d), not by run seed
    lam = make_lambda(epsilon, L_TARGET) # lambda depends on epsilon only

    rng_init = np.random.default_rng(seed)
    x = sample_ball(1, d, R, rng_init)[0]
    rng = np.random.default_rng(seed + 10_000)

    t                  = 0
    total_samples      = 0
    path_length        = 0.0
    converged          = False
    projections_active = 0
    coord_traj         = []
    activation_step    = -1

    def _record_coord(step):
        # x in ball of radius 0.5 => |x_i| <= 0.5 < 1 => rho(x) = x
        lc = float((U.T @ x)[T - 1])
        coord_traj.append((step, lc))
        return lc

    # Initial stationarity + first coord record
    last_stat      = stat_estimate(x, U, lam, L_TARGET, SIGMA, rng)
    total_samples += 200
    _record_coord(0)

    if last_stat <= epsilon:
        _save_trajectory(T, epsilon, seed, coord_traj)
        return dict(T=T, d=d, schedule=SCHEDULE, epsilon=epsilon, seed=seed,
                    N=total_samples, path_length=path_length,
                    success=1, budget_limited=0,
                    projection_rate=0.0, last_coord_activation_step=-1)

    while total_samples < N_MAX:
        t += 1

        # Schedule B: eta_t = min(1/L, 1/sqrt(t)), batch_size = 1
        eta        = min(1.0 / L_TARGET, 1.0 / np.sqrt(t))
        batch_size = 1

        g           = stoch_grad(x, U, lam, L_TARGET, batch_size, SIGMA, rng)
        total_samples += batch_size

        # Diagnostic 1: projection activity
        x_raw = x - eta * g
        if np.linalg.norm(x_raw) > R:
            projections_active += 1

        x_new       = project(x_raw)
        path_length += float(np.linalg.norm(x_new - x))
        x           = x_new

        if t % EVAL_EVERY == 0:
            # Diagnostics 2 & 3
            lc = _record_coord(t)
            if activation_step == -1 and abs(lc) > 0.01:
                activation_step = t

            last_stat      = stat_estimate(x, U, lam, L_TARGET, SIGMA, rng)
            total_samples += 200
            if last_stat <= epsilon:
                converged = True
                break

    _save_trajectory(T, epsilon, seed, coord_traj)

    return dict(T=T, d=d, schedule=SCHEDULE, epsilon=epsilon, seed=seed,
                N=total_samples, path_length=path_length,
                success=int(converged), budget_limited=int(not converged),
                projection_rate=projections_active / t if t > 0 else 0.0,
                last_coord_activation_step=activation_step)


# ---------------------------------------------------------------------------
# Runtime estimator: time the T=10 block and project forward
# ---------------------------------------------------------------------------

def estimate_and_run_T10(writer, f, done: int) -> tuple[int, float]:
    """
    Run all T=10 configurations, measure wall time, print runtime estimate
    for remaining T values, then return (done_count, seconds_per_run_T10).
    """
    t0 = time.perf_counter()
    n_T10 = len(EPS_GRID) * N_SEEDS

    for epsilon, seed in product(EPS_GRID, range(N_SEEDS)):
        result = run_single_T(10, epsilon, seed)
        writer.writerow(result)
        f.flush()
        done += 1
        if done % 5 == 0 or done == TOTAL_RUNS:
            status = 'OK' if result['success'] else 'BUDGET'
            print(f"  {done}/{TOTAL_RUNS}  (T=10, eps={epsilon}, seed={seed})  "
                  f"N={result['N']:,}  {status}")

    elapsed_T10 = time.perf_counter() - t0
    sec_per_run = elapsed_T10 / n_T10

    print(f"\n--- Runtime estimate (based on {n_T10} T=10 runs, "
          f"{elapsed_T10:.1f}s total, {sec_per_run:.1f}s/run) ---")
    # Each SGD step costs O(T*d) = O(T * 2T) = O(T^2); runs hitting N_MAX
    # scale proportionally. Use T^2 / 10^2 as the scaling factor.
    remaining_est = 0.0
    for T in T_VALUES[1:]:
        scale      = (T / 10.0) ** 2
        est_secs   = sec_per_run * len(EPS_GRID) * N_SEEDS * scale
        remaining_est += est_secs
        print(f"  T={T:>3}  ({len(EPS_GRID)*N_SEEDS} runs)  "
              f"~{est_secs/60:.1f} min  (scale factor {scale:.0f}x)")
    print(f"  Total remaining estimate: ~{remaining_est/60:.0f} min")
    print(f"  Grand total estimate:     ~{(elapsed_T10 + remaining_est)/60:.0f} min\n")

    return done, sec_per_run


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    fieldnames = ['T', 'd', 'schedule', 'epsilon', 'seed', 'N', 'path_length',
                  'success', 'budget_limited', 'projection_rate',
                  'last_coord_activation_step']
    already_done = set()

    os.makedirs('results', exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                already_done.add((row['T'], row['epsilon'], row['seed']))
        mode = 'a'
        print(f"Resuming: {len(already_done)} results already saved.")
    else:
        mode = 'w'

    done = len(already_done)

    with open(RESULTS_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        # --- T=10 block with runtime estimation ---
        t10_done = sum(
            1 for (T, eps, s) in already_done if T == '10'
        )
        if t10_done < len(EPS_GRID) * N_SEEDS:
            done, _ = estimate_and_run_T10(writer, f, done)
        else:
            print("  T=10 block already complete, skipping estimation.")

        # --- Remaining T values ---
        for T, epsilon, seed in product(T_VALUES[1:], EPS_GRID, range(N_SEEDS)):
            if (str(T), str(epsilon), str(seed)) in already_done:
                done += 1
                continue

            result = run_single_T(T, epsilon, seed)
            writer.writerow(result)
            f.flush()
            done += 1

            status = 'OK' if result['success'] else 'BUDGET'
            if done % 5 == 0 or done == TOTAL_RUNS:
                print(f"  {done}/{TOTAL_RUNS}  (T={T}, eps={epsilon}, seed={seed})  "
                      f"N={result['N']:,}  {status}")

    print(f"\nDone. Saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Quick analysis
# ---------------------------------------------------------------------------

def analyze():
    rows = []
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            rows.append(dict(
                T=int(row['T']), d=int(row['d']),
                epsilon=float(row['epsilon']), seed=int(row['seed']),
                N=int(row['N']), success=int(row['success']),
                budget_limited=int(row['budget_limited']),
                projection_rate=float(row['projection_rate']),
                last_coord_activation_step=int(row['last_coord_activation_step']),
            ))

    print("\n=== T-SWEEP RESULTS ===")
    print(f"{'T':>5}  {'eps':>6}  {'succ':>5}  {'med_N':>10}  "
          f"{'proj_rate':>10}  {'act_step':>10}")
    print("-" * 58)
    for T in T_VALUES:
        for eps in EPS_GRID:
            sub = [r for r in rows if r['T'] == T and r['epsilon'] == eps]
            if not sub:
                continue
            sr     = np.mean([r['success'] for r in sub])
            med_N  = int(np.median([r['N'] for r in sub]))
            med_pr = np.median([r['projection_rate'] for r in sub])
            acts   = [r['last_coord_activation_step'] for r in sub]
            act_str = str(int(np.median([a for a in acts if a != -1]))) \
                      if any(a != -1 for a in acts) else '-1'
            print(f"{T:>5}  {eps:>6.2f}  {sr:>5.0%}  {med_N:>10,}  "
                  f"{med_pr:>10.3f}  {act_str:>10}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Arjevani T-Sweep Experiment ===")
    print(f"T values   : {T_VALUES}")
    print(f"d values   : {[2*T for T in T_VALUES]}")
    print(f"eps grid   : {EPS_GRID}")
    print(f"schedule   : {SCHEDULE}")
    print(f"seeds/cfg  : {N_SEEDS}")
    print(f"N_max      : {N_MAX:,}")
    print(f"total runs : {TOTAL_RUNS}\n")

    run_experiment()
    analyze()
