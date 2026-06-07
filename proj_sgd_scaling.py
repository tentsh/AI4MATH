"""
Proj-SGD scaling exponent experiment (fixed version).
Fixes applied:
  1. Analytic L — no numerical calibration
  2. Consistent noise: sigma/sqrt(batch_size) everywhere
  3. Stochastic stationarity metric (batch 200)
  4. No filtering — report full and clean slopes side-by-side
  5. Stop on stochastic stationarity every 200 steps
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
from itertools import product

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
R   = 0.5   # ball radius
D   = 10    # dimension

def project(x):
    norm = np.linalg.norm(x)
    return x * (R / norm) if norm > R else x.copy()

# ---------------------------------------------------------------------------
# Fix 1: Analytic Lipschitz constants — never estimated from data
#
# Zero-chain: F(x) = 0.5||x||^2 + (0.1/d)*sum_{k=0}^{d-2} exp(-x_k^2/0.1)*x_{k+1}^2
#   Hessian diagonal entries bounded by 1 + (0.1/d)*[20 + 400*x_k^2]*x_{k+1}^2
#   On the ball (|x_k|<=0.5): max entry <= 1 + (0.1/d)*120*0.25 = 1 + 3/d < 2   => L=2
#
# Sinusoid: F(x) = 0.5||x||^2 + (0.1/d)*sum_k sin(x_k)
#   H = I + diag(-(0.1/d)*sin(x_k)), ||H||_2 <= 1 + 0.1/d < 1.01              => L=1
# ---------------------------------------------------------------------------
L_BY_OBJ = {
    'zerochain': 2.0,
    'sinusoid':  1.0,
}

def print_analytic_L():
    print("=== Analytic Lipschitz constants (Fix 1) ===")
    print(f"  zero-chain : L = {L_BY_OBJ['zerochain']}")
    print(f"    F = 0.5||x||^2 + (0.1/d)*sum_k exp(-x_k^2/0.1)*x_{{k+1}}^2")
    print(f"    Hessian bound on B(0,{R}): max eigenvalue <= 1 + 3/d < 2  =>  L=2")
    print(f"  sinusoid   : L = {L_BY_OBJ['sinusoid']}")
    print(f"    F = 0.5||x||^2 + (0.1/d)*sum_k sin(x_k)")
    print(f"    Hessian = I - diag((0.1/d)*sin(x_k)), ||H||_2 <= 1+0.1/d < 1.01  =>  L=1")
    print()

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def f_zerochain(x):
    val = 0.5 * np.dot(x, x)
    c = 0.1 / D
    for k in range(D - 1):
        val += c * np.exp(-x[k]**2 / 0.1) * x[k + 1]**2
    return val

def grad_zerochain(x):
    g = x.copy()
    c = 0.1 / D
    for k in range(D - 1):
        ek = np.exp(-x[k]**2 / 0.1)
        g[k]     += c * (-2.0 * x[k] / 0.1) * ek * x[k + 1]**2
        g[k + 1] += c * 2.0 * x[k + 1] * ek
    return g

def f_sinusoid(x):
    return 0.5 * np.dot(x, x) + (0.1 / D) * np.sum(np.sin(x))

def grad_sinusoid(x):
    return x + (0.1 / D) * np.cos(x)

# (f, grad, L)
OBJECTIVES = {
    'zerochain': (f_zerochain, grad_zerochain, L_BY_OBJ['zerochain']),
    'sinusoid':  (f_sinusoid,  grad_sinusoid,  L_BY_OBJ['sinusoid']),
}

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
SCHEDULES = ['A', 'B', 'C', 'D']

def get_eta(schedule, t, L):
    if schedule == 'A':
        return 1.0 / L
    if schedule == 'B':
        return min(1.0 / L, 1.0 / np.sqrt(max(t, 1)))
    if schedule == 'C':
        return 1.0 / (L * max(t, 1) ** (1.0 / 3.0))
    if schedule == 'D':
        return 1.0 / L
    raise ValueError(schedule)

def get_batch(schedule, t):
    if schedule in ('A', 'B', 'C'):
        return 1
    if schedule == 'D':
        return int(np.ceil(np.sqrt(max(t, 1))))
    raise ValueError(schedule)

# ---------------------------------------------------------------------------
# Fix 2: Consistent noise model — single formula, sigma/sqrt(batch_size)
# ---------------------------------------------------------------------------
SIGMA = 0.5

def stoch_grad(grad_fn, x, batch_size, rng):
    return grad_fn(x) + rng.standard_normal(D) * SIGMA / np.sqrt(batch_size)

# ---------------------------------------------------------------------------
# Fix 3: Stochastic stationarity metric
#   batch_grad = grad(x) + N(0, sigma^2/EVAL_BATCH * I)
#   stat = ||x - P_X(x - (1/L)*batch_grad)|| * L
# ---------------------------------------------------------------------------
EVAL_BATCH = 200

def stoch_stationarity(grad_fn, x, L, rng):
    batch_grad = grad_fn(x) + rng.standard_normal(D) * SIGMA / np.sqrt(EVAL_BATCH)
    x_new = project(x - (1.0 / L) * batch_grad)
    return np.linalg.norm(x_new - x) * L

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
EPS_GRID   = [0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]
N_SEEDS    = 10
N_MAX      = 1_000_000
EVAL_EVERY = 200          # Fix 5: evaluate every 200 steps
RESULTS_FILE = 'results/results_clean.csv'

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(obj_name, schedule, eps, seed):
    _, grad_fn, L = OBJECTIVES[obj_name]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(D)
    x = project(x)
    path_length = 0.0
    t = 1

    while t <= N_MAX:
        # Fix 5: check every EVAL_EVERY steps using stochastic stationarity
        if t == 1 or t % EVAL_EVERY == 1:
            if stoch_stationarity(grad_fn, x, L, rng) <= eps:
                return t, path_length, True

        batch = get_batch(schedule, t)
        eta   = get_eta(schedule, t, L)
        g     = stoch_grad(grad_fn, x, batch, rng)
        x_new = project(x - eta * g)
        path_length += np.linalg.norm(x_new - x)
        x = x_new
        t += 1

    return N_MAX, path_length, False

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment():
    fieldnames = ['objective', 'schedule', 'epsilon', 'seed', 'N', 'path_length', 'success']
    already_done = set()

    os.makedirs('results', exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                already_done.add((row['objective'], row['schedule'],
                                   row['epsilon'], row['seed']))
        mode = 'a'
        print(f"Resuming: {len(already_done)} results already saved.")
    else:
        mode = 'w'

    total = len(OBJECTIVES) * len(SCHEDULES) * len(EPS_GRID) * N_SEEDS
    done  = len(already_done)

    with open(RESULTS_FILE, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        for obj_name, schedule, eps in product(OBJECTIVES, SCHEDULES, EPS_GRID):
            group = []
            for seed in range(N_SEEDS):
                if (obj_name, schedule, str(eps), str(seed)) in already_done:
                    continue
                N, pl, ok = run_single(obj_name, schedule, eps, seed)
                group.append((seed, N, pl, ok))

            if not group:
                continue

            n_maxed = sum(1 for _, N, _, ok in group if not ok)
            budget_limited = n_maxed > len(group) / 2

            for seed, N, pl, ok in group:
                writer.writerow({
                    'objective':   obj_name,
                    'schedule':    schedule,
                    'epsilon':     eps,
                    'seed':        seed,
                    'N':           N,
                    'path_length': pl,
                    'success':     'budget_limited' if (budget_limited and not ok) else int(ok),
                })

            f.flush()
            done += len(group)
            bl = ' [budget_limited]' if budget_limited else ''
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}  ({obj_name}, {schedule}, eps={eps}){bl}")

    print(f"Done. Results saved to {RESULTS_FILE}")

# ---------------------------------------------------------------------------
# Analysis — Fix 4: no filtering; report full and clean slopes
# ---------------------------------------------------------------------------
def bootstrap_slope(lx, ly, n_boot=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(lx)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_, y_ = lx[idx], ly[idx]
        if np.ptp(x_) < 1e-10:
            continue
        slopes.append(np.polyfit(x_, y_, 1)[0])
    slopes = np.array(slopes)
    return np.mean(slopes), np.percentile(slopes, 2.5), np.percentile(slopes, 97.5)

def load_results():
    rows = []
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            s = row['success']
            rows.append({
                'objective':    row['objective'],
                'schedule':     row['schedule'],
                'epsilon':      float(row['epsilon']),
                'seed':         int(row['seed']),
                'N':            int(row['N']),
                'path_length':  float(row['path_length']),
                'success':      1 if s == '1' else 0,
                'hit_nmax':     int(row['N']) >= N_MAX,
            })
    return rows

def analyze():
    rows = load_results()
    rng  = np.random.default_rng(0)

    print("\n=== SCALING EXPONENT ANALYSIS (N vs 1/eps) ===")
    hdr = f"{'Obj':<12}{'Sched':<6}  {'p_full':>8}  {'95%CI_full':<18}  {'p_clean':>8}  {'95%CI_clean':<18}"
    print(hdr)
    print("-" * len(hdr))

    summary = {}

    for obj_name in OBJECTIVES:
        for schedule in SCHEDULES:
            sub = [r for r in rows
                   if r['objective'] == obj_name and r['schedule'] == schedule]
            if not sub:
                continue

            full_pts, clean_pts = [], []
            for eps in EPS_GRID:
                ep_rows = [r for r in sub if r['epsilon'] == eps]
                if not ep_rows:
                    continue
                frac_maxed = np.mean([r['hit_nmax'] for r in ep_rows])
                lx = np.log(1.0 / eps)
                for r in ep_rows:
                    ly = np.log(max(r['N'], 1))
                    full_pts.append((lx, ly))
                    if frac_maxed < 0.30:          # clean: <30% seeds hit N_MAX
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
                    return f"{'n/a':>8}", f"{'n/a':<18}"
                m, lo, hi = r
                return f"{m:>8.3f}", f"[{lo:.3f}, {hi:.3f}]"

            sf, cf = fmt(rf)
            sc, cc = fmt(rc)
            print(f"{obj_name:<12}{schedule:<6}  {sf}  {cf}  {sc}  {cc}")

            summary[(obj_name, schedule)] = {
                'full': full_pts, 'clean': clean_pts,
                'fit_full': rf,   'fit_clean': rc,
            }

    plot_results(rows, summary)

def plot_results(rows, summary):
    obj_names = list(OBJECTIVES)
    colors  = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
    markers = {'A': 'o',       'B': 's',        'C': '^',        'D': 'D'}

    fig, axes = plt.subplots(2, len(obj_names), figsize=(6 * len(obj_names), 10))
    if len(obj_names) == 1:
        axes = axes.reshape(2, 1)

    for col, obj_name in enumerate(obj_names):
        ax_N  = axes[0, col]
        ax_PL = axes[1, col]

        for sched in SCHEDULES:
            sub = [r for r in rows
                   if r['objective'] == obj_name and r['schedule'] == sched]
            if not sub:
                continue

            inv_eps, med_N, med_PL = [], [], []
            for eps in sorted(EPS_GRID):
                ep_rows = [r for r in sub if r['epsilon'] == eps]
                if not ep_rows:
                    continue
                inv_eps.append(1.0 / eps)
                med_N.append(np.median([r['N'] for r in ep_rows]))
                med_PL.append(np.median([r['path_length'] for r in ep_rows]))

            ax_N.loglog(inv_eps, med_N,  color=colors[sched],
                        marker=markers[sched], label=f"Sched {sched}", linewidth=1.5)
            ax_PL.loglog(inv_eps, med_PL, color=colors[sched],
                         marker=markers[sched], label=f"Sched {sched}", linewidth=1.5)

            # Overlay fitted slopes: solid=full, dashed=clean
            key = (obj_name, sched)
            if key not in summary:
                continue
            for pts_key, ls in [('full', '-'), ('clean', '--')]:
                pts = summary[key][pts_key]
                if len(pts) < 4:
                    continue
                lx = np.array([p[0] for p in pts])
                ly = np.array([p[1] for p in pts])
                if np.ptp(lx) < 1e-10:
                    continue
                slope, intercept = np.polyfit(lx, ly, 1)
                xf = np.linspace(lx.min(), lx.max(), 50)
                yf = slope * xf + intercept
                ax_N.loglog(np.exp(xf), np.exp(yf), ls,
                            color=colors[sched], alpha=0.45, linewidth=1.2)

        for ax, ylabel, title in [
            (ax_N,  'Iterations $N$ (median)',      f'{obj_name}: $N(\\varepsilon)$ vs $1/\\varepsilon$'),
            (ax_PL, 'Path length $\\ell_N$ (median)', f'{obj_name}: path length vs $1/\\varepsilon$'),
        ]:
            ax.set_title(title)
            ax.set_xlabel('$1/\\varepsilon$')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(
        'Proj-SGD scaling: $N(\\varepsilon)\\sim\\varepsilon^{-p}$\n'
        'Solid fit lines = full data  |  Dashed = clean (<30% seeds at $N_{\\max}$)',
        fontsize=12,
    )
    plt.tight_layout()
    out = 'figures/scaling_results_clean.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    print_analytic_L()
    print(f"N_MAX={N_MAX:,}  SIGMA={SIGMA}  EVAL_EVERY={EVAL_EVERY}  EVAL_BATCH={EVAL_BATCH}")
    print(f"Objectives : {list(OBJECTIVES)}")
    print(f"Schedules  : {SCHEDULES}")
    print(f"Eps grid   : {EPS_GRID}")
    print(f"Seeds/config: {N_SEEDS}\n")

    run_experiment()
    analyze()
