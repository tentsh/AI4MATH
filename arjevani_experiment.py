"""
Full Arjevani zero-chain scaling experiment.
Builds on arjevani_sgd.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
from itertools import product

from arjevani_rotated import make_U
from arjevani_full import F_scaled, grad_F_scaled, make_lambda, sample_ball
from arjevani_sgd import project, stoch_grad, stat_estimate

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
L_TARGET   = 1.0
D_X        = 1.0
SIGMA      = 1.0
ELL_1      = 152
ETA_0      = 0.2
EPS_GRID   = [0.3, 0.2, 0.15, 0.1, 0.07]
N_SEEDS    = 5
N_MAX      = 500_000
EVAL_EVERY = 500
R          = D_X / 2          # ball radius = 0.5
SCHEDULES  = ['B', 'D']
RESULTS_FILE = 'results_arjevani.csv'


def compute_T(eps: float) -> int:
    return max(5, min(50, int(round(1.0 / (eps * eps * 4)))))


def print_params():
    print("=== Arjevani Experiment Parameters ===")
    print(f"  L_target={L_TARGET}  D_X={D_X}  sigma={SIGMA}  "
          f"ell_1={ELL_1}  eta_0={ETA_0}")
    print(f"  N_max={N_MAX:,}  seeds={N_SEEDS}  schedules={SCHEDULES}")
    print(f"\n  {'eps':<6}  {'T':>4}  {'d':>4}")
    print(f"  {'-'*18}")
    for eps in EPS_GRID:
        T = compute_T(eps)
        print(f"  {eps:<6.2f}  {T:>4}  {2*T:>4}")
    print()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(schedule: str, epsilon: float, seed: int) -> dict:
    T   = compute_T(epsilon)
    d   = 2 * T
    U   = make_U(d, T, seed=0)          # U fixed by problem, not by run seed
    lam = make_lambda(epsilon, L_TARGET)

    rng_init = np.random.default_rng(seed)
    x = sample_ball(1, d, R, rng_init)[0]
    rng = np.random.default_rng(seed + 10_000)

    t             = 0
    total_samples = 0
    path_length   = 0.0
    converged     = False
    last_stat     = float('inf')

    # Initial stationarity estimate
    last_stat      = stat_estimate(x, U, lam, L_TARGET, SIGMA, rng)
    total_samples += 200
    if last_stat <= epsilon:
        return dict(schedule=schedule, epsilon=epsilon, seed=seed,
                    T=T, d=d, N=total_samples, path_length=path_length,
                    success=1, budget_limited=0)

    while total_samples < N_MAX:
        t += 1

        if schedule == 'B':
            eta        = min(1.0 / L_TARGET, 1.0 / np.sqrt(t))
            batch_size = 1
        else:   # D
            eta        = 1.0 / L_TARGET
            batch_size = int(np.ceil(np.sqrt(t)))

        g      = stoch_grad(x, U, lam, L_TARGET, batch_size, SIGMA, rng)
        total_samples += batch_size

        x_new       = project(x - eta * g)
        path_length += float(np.linalg.norm(x_new - x))
        x           = x_new

        if t % EVAL_EVERY == 0:
            last_stat      = stat_estimate(x, U, lam, L_TARGET, SIGMA, rng)
            total_samples += 200
            if last_stat <= epsilon:
                converged = True
                break

    budget_limited = int(not converged)
    return dict(schedule=schedule, epsilon=epsilon, seed=seed,
                T=T, d=d, N=total_samples, path_length=path_length,
                success=int(converged), budget_limited=budget_limited)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    fieldnames = ['schedule', 'epsilon', 'seed', 'T', 'd',
                  'N', 'path_length', 'success', 'budget_limited']
    already_done = set()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                already_done.add((row['schedule'], row['epsilon'], row['seed']))
        mode = 'a'
        print(f"Resuming: {len(already_done)} results already saved.")
    else:
        mode = 'w'

    total_runs = len(SCHEDULES) * len(EPS_GRID) * N_SEEDS
    done       = len(already_done)

    with open(RESULTS_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        for schedule, epsilon, seed in product(SCHEDULES, EPS_GRID, range(N_SEEDS)):
            if (schedule, str(epsilon), str(seed)) in already_done:
                done += 1
                continue

            result = run_single(schedule, epsilon, seed)
            writer.writerow(result)
            f.flush()
            done += 1

            status = 'OK' if result['success'] else 'BUDGET'
            if done % 10 == 0 or done == total_runs:
                print(f"  {done}/{total_runs}  "
                      f"({schedule}, eps={epsilon}, seed={seed})  "
                      f"N={result['N']:,}  {status}")

    print(f"\nDone. Saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_results() -> list:
    rows = []
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            rows.append(dict(
                schedule=row['schedule'],
                epsilon=float(row['epsilon']),
                seed=int(row['seed']),
                T=int(row['T']),
                d=int(row['d']),
                N=int(row['N']),
                path_length=float(row['path_length']),
                success=int(row['success']),
                budget_limited=int(row['budget_limited']),
            ))
    return rows


def bootstrap_slope(lx: np.ndarray, ly: np.ndarray,
                    n_boot: int = 1000,
                    rng: np.random.Generator = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(lx)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_, y_ = lx[idx], ly[idx]
        if np.ptp(x_) < 1e-10:
            continue
        slopes.append(np.polyfit(x_, y_, 1)[0])
    slopes = np.array(slopes)
    return float(np.mean(slopes)), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def analyze():
    rows = load_results()
    rng  = np.random.default_rng(42)

    print("\n=== SCALING ANALYSIS: log N vs log(1/epsilon) ===")
    hdr = (f"{'Sched':<6}  {'p_full':>8}  {'CI_full':<20}"
           f"  {'p_clean':>8}  {'CI_clean':<20}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}

    for schedule in SCHEDULES:
        sub = [r for r in rows if r['schedule'] == schedule]

        full_pts, clean_pts = [], []
        for eps in EPS_GRID:
            ep_rows = [r for r in sub if r['epsilon'] == eps]
            if not ep_rows:
                continue
            success_rate = np.mean([r['success'] for r in ep_rows])
            lx = np.log(1.0 / eps)
            for r in ep_rows:
                ly = np.log(max(r['N'], 1))
                full_pts.append((lx, ly))
                if success_rate >= 0.60:
                    clean_pts.append((lx, ly))

        def fit(pts):
            if len(pts) < 4:
                return None
            lx = np.array([p[0] for p in pts])
            ly = np.array([p[1] for p in pts])
            if np.ptp(lx) < 1e-10:
                return None
            return bootstrap_slope(lx, ly, rng=rng)

        rf = fit(full_pts)
        rc = fit(clean_pts)

        def fmt(r):
            if r is None:
                return f"{'n/a':>8}", f"{'n/a':<20}"
            m, lo, hi = r
            return f"{m:>8.3f}", f"[{lo:.3f}, {hi:.3f}]"

        sf, cf = fmt(rf)
        sc, cc = fmt(rc)
        print(f"{schedule:<6}  {sf}  {cf}  {sc}  {cc}")

        # Per-epsilon success summary
        print(f"         success rates by eps: ", end='')
        for eps in EPS_GRID:
            ep_rows = [r for r in sub if r['epsilon'] == eps]
            sr = np.mean([r['success'] for r in ep_rows]) if ep_rows else float('nan')
            print(f"{eps}:{sr:.0%}  ", end='')
        print()

        summary[schedule] = {'full': full_pts, 'clean': clean_pts,
                             'fit_full': rf, 'fit_clean': rc}

    plot_results(rows, summary)


def plot_results(rows: list, summary: dict):
    colors = {'B': '#1f77b4', 'D': '#d62728'}
    markers = {'B': 'o', 'D': 's'}

    fig, ax = plt.subplots(figsize=(7, 5))

    for schedule in SCHEDULES:
        sub = [r for r in rows if r['schedule'] == schedule]
        inv_eps, med_N = [], []
        for eps in sorted(EPS_GRID):
            ep_rows = [r for r in sub if r['epsilon'] == eps]
            if not ep_rows:
                continue
            inv_eps.append(1.0 / eps)
            med_N.append(float(np.median([r['N'] for r in ep_rows])))

        ax.loglog(inv_eps, med_N, color=colors[schedule],
                  marker=markers[schedule], linewidth=1.8, markersize=7,
                  label=f'Schedule {schedule} (median N)')

        # Overlay fitted slopes (full=solid, clean=dashed)
        info = summary.get(schedule, {})
        for pts_key, ls, alpha in [('full', '-', 0.4), ('clean', '--', 0.6)]:
            pts = info.get(pts_key, [])
            if len(pts) < 4:
                continue
            lx = np.array([p[0] for p in pts])
            ly = np.array([p[1] for p in pts])
            if np.ptp(lx) < 1e-10:
                continue
            slope, intercept = np.polyfit(lx, ly, 1)
            xf = np.linspace(lx.min(), lx.max(), 50)
            yf = slope * xf + intercept
            ax.loglog(np.exp(xf), np.exp(yf), ls,
                      color=colors[schedule], alpha=alpha, linewidth=1.2)

    # Reference lines at slope 3 and slope 4
    x_ref = np.array([1.0 / max(EPS_GRID), 1.0 / min(EPS_GRID)])
    for slope_ref, ls, label in [(3, ':', 'slope 3'), (4, ':', 'slope 4')]:
        y_ref = x_ref ** slope_ref
        # Shift vertically so the line passes through the plot center
        med_all = np.median([r['N'] for r in rows])
        x_mid   = np.sqrt(x_ref[0] * x_ref[1])
        shift   = med_all / x_mid**slope_ref
        ax.loglog(x_ref, y_ref * shift, ls, color='gray',
                  linewidth=1.0, alpha=0.7, label=label)

    ax.set_xlabel('$1/\\varepsilon$', fontsize=12)
    ax.set_ylabel('Samples $N$ (median)', fontsize=12)
    ax.set_title('Arjevani zero-chain: $N(\\varepsilon) \\sim \\varepsilon^{-p}$\n'
                 'Solid fit = full data  |  Dashed = clean ($\\geq$60% converged)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = 'arjevani_scaling.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    print_params()
    run_experiment()
    analyze()
