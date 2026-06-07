"""
Analysis of results_t_sweep.csv: three diagnostic plots and summary table.
Can be run on partial data; missing (T, epsilon) cells are skipped.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
from collections import defaultdict

RESULTS_FILE = 'results/results_t_sweep.csv'
T_VALUES     = [10, 25, 50, 100]
EPS_GRID     = [0.2, 0.15, 0.1]
N_MAX        = 1_000_000


# ---------------------------------------------------------------------------
# Load and deduplicate
# ---------------------------------------------------------------------------

def load() -> list[dict]:
    rows = []
    seen = set()
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            key = (row['T'], row['epsilon'], row['seed'])
            if key in seen:
                continue                # drop duplicates
            seen.add(key)
            rows.append(dict(
                T=int(row['T']),
                d=int(row['d']),
                epsilon=float(row['epsilon']),
                seed=int(row['seed']),
                N=int(row['N']),
                success=int(row['success']),
                budget_limited=int(row['budget_limited']),
                projection_rate=float(row['projection_rate']),
                last_coord_activation_step=int(row['last_coord_activation_step']),
            ))
    return rows


# ---------------------------------------------------------------------------
# Bootstrap slope helper
# ---------------------------------------------------------------------------

def bootstrap_slope(lx, ly, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(lx)
    if n < 3 or np.ptp(lx) < 1e-10:
        return np.nan, np.nan, np.nan
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.ptp(lx[idx]) < 1e-10:
            continue
        slopes.append(np.polyfit(lx[idx], ly[idx], 1)[0])
    slopes = np.array(slopes)
    return float(np.mean(slopes)), float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(rows):
    print("\n=== SUMMARY TABLE ===")
    print(f"{'T':>5}  {'eps':>6}  {'succ%':>6}  {'med_N':>10}  "
          f"{'proj_rate':>10}  {'act_step':>10}  {'n_runs':>7}")
    print("-" * 68)
    table = {}
    for T in T_VALUES:
        for eps in EPS_GRID:
            sub = [r for r in rows if r['T'] == T and r['epsilon'] == eps]
            if not sub:
                continue
            sr    = float(np.mean([r['success'] for r in sub]))
            med_N = int(np.median([r['N'] for r in sub]))
            # projection_rate: mean over succeeded runs only
            ok    = [r for r in sub if not r['budget_limited']]
            mpr   = float(np.mean([r['projection_rate'] for r in ok])) if ok else float('nan')
            # activation: treat -1 as N_MAX
            acts  = [r['last_coord_activation_step'] if r['last_coord_activation_step'] != -1
                     else N_MAX for r in sub]
            med_act = int(np.median(acts))
            print(f"{T:>5}  {eps:>6.2f}  {sr:>6.0%}  {med_N:>10,}  "
                  f"{mpr:>10.3f}  {med_act:>10,}  {len(sub):>7}")
            table[(T, eps)] = dict(sr=sr, med_N=med_N, mpr=mpr, med_act=med_act)
    return table


# ---------------------------------------------------------------------------
# Plot 1: fitted slope vs T
# ---------------------------------------------------------------------------

def plot_slope_vs_T(rows, rng):
    slopes, lo_ci, hi_ci, T_vals_valid = [], [], [], []

    for T in T_VALUES:
        sub_T = [r for r in rows if r['T'] == T]
        pts   = []
        for eps in EPS_GRID:
            ep_rows = [r for r in sub_T if r['epsilon'] == eps]
            if not ep_rows:
                continue
            sr = np.mean([r['success'] for r in ep_rows])
            if sr < 0.50:
                continue          # skip cells below 50% success
            lx = np.log(1.0 / eps)
            for r in ep_rows:
                pts.append((lx, np.log(max(r['N'], 1))))

        if len(pts) < 4:
            print(f"  T={T}: insufficient clean data for slope fit ({len(pts)} pts)")
            continue

        lx = np.array([p[0] for p in pts])
        ly = np.array([p[1] for p in pts])
        m, lo, hi = bootstrap_slope(lx, ly, rng=rng)
        slopes.append(m); lo_ci.append(lo); hi_ci.append(hi)
        T_vals_valid.append(T)
        print(f"  T={T:>3}: slope={m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]"
              f"  ({len(pts)} data pts)")

    if not T_vals_valid:
        print("  No valid T values for slope plot.")
        return slopes

    fig, ax = plt.subplots(figsize=(7, 5))
    err_lo = np.array(slopes) - np.array(lo_ci)
    err_hi = np.array(hi_ci)  - np.array(slopes)
    ax.errorbar(T_vals_valid, slopes,
                yerr=[err_lo, err_hi],
                fmt='o-', color='#1f77b4', linewidth=1.8,
                markersize=7, capsize=5, label='fitted slope p')
    for ref, col in [(3, '#aec7e8'), (4, '#ffbb78')]:
        ax.axhline(ref, linestyle='--', color=col, linewidth=1.2,
                   label=f'slope = {ref}')
    ax.set_xlabel('T  (chain length)', fontsize=12)
    ax.set_ylabel('Fitted exponent $p$  ($N \\sim \\varepsilon^{-p}$)', fontsize=12)
    ax.set_title('Scaling exponent vs chain length T\n'
                 '(eps points with $\\geq$50% success only; 95% bootstrap CI)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/t_sweep_slope_vs_T.png', dpi=150, bbox_inches='tight')
    print("  Saved figures/t_sweep_slope_vs_T.png")
    return slopes


# ---------------------------------------------------------------------------
# Plot 2: projection activity vs T
# ---------------------------------------------------------------------------

def plot_projection_vs_T(rows):
    colors = {0.2: '#1f77b4', 0.15: '#ff7f0e', 0.1: '#2ca02c'}
    fig, ax = plt.subplots(figsize=(7, 5))

    for eps in EPS_GRID:
        T_pts, pr_pts = [], []
        for T in T_VALUES:
            sub = [r for r in rows
                   if r['T'] == T and r['epsilon'] == eps and not r['budget_limited']]
            if not sub:
                continue
            T_pts.append(T)
            pr_pts.append(float(np.mean([r['projection_rate'] for r in sub])))
        if T_pts:
            ax.plot(T_pts, pr_pts, 'o-', color=colors[eps], linewidth=1.8,
                    markersize=7, label=f'eps={eps}')

    ax.set_xlabel('T  (chain length)', fontsize=12)
    ax.set_ylabel('Mean projection rate\n(converged runs only)', fontsize=12)
    ax.set_title('Projection activity vs chain length T', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/t_sweep_projection_vs_T.png', dpi=150, bbox_inches='tight')
    print("  Saved figures/t_sweep_projection_vs_T.png")


# ---------------------------------------------------------------------------
# Plot 3: last-coord activation step vs T (log-log)
# ---------------------------------------------------------------------------

def plot_activation_vs_T(rows):
    colors = {0.2: '#1f77b4', 0.15: '#ff7f0e', 0.1: '#2ca02c'}
    fig, ax = plt.subplots(figsize=(7, 5))

    for eps in EPS_GRID:
        T_pts, act_pts = [], []
        for T in T_VALUES:
            sub = [r for r in rows if r['T'] == T and r['epsilon'] == eps]
            if not sub:
                continue
            acts = [r['last_coord_activation_step'] if r['last_coord_activation_step'] != -1
                    else N_MAX for r in sub]
            T_pts.append(T)
            act_pts.append(float(np.median(acts)))
        if T_pts:
            ax.loglog(T_pts, act_pts, 'o-', color=colors[eps], linewidth=1.8,
                      markersize=7, label=f'eps={eps}')

    ax.set_xlabel('T  (chain length)', fontsize=12)
    ax.set_ylabel('Median last-coord activation step\n(-1 → N_max)', fontsize=12)
    ax.set_title('Last-coordinate activation step vs T\n'
                 '(log-log; -1 treated as N_max)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/t_sweep_activation_vs_T.png', dpi=150, bbox_inches='tight')
    print("  Saved figures/t_sweep_activation_vs_T.png")


# ---------------------------------------------------------------------------
# Diagnostic signature checks
# ---------------------------------------------------------------------------

def check_signatures(rows, slopes, table):
    print("\n=== DIAGNOSTIC SIGNATURES ===")

    # 1. Monotone slope increase with T
    if len(slopes) >= 2 and all(slopes[i] < slopes[i+1] for i in range(len(slopes)-1)):
        print("  MONOTONE SLOPE INCREASE")
    else:
        diffs = [f"{slopes[i+1]-slopes[i]:+.3f}" for i in range(len(slopes)-1)]
        print(f"  slope not strictly monotone in T  (diffs: {diffs})")

    # 2. Monotone projection_rate increase with T
    # Use eps=0.1 (hardest) as the probe curve
    pr_by_T = {}
    for T in T_VALUES:
        sub = [r for r in rows
               if r['T'] == T and r['epsilon'] == 0.1 and not r['budget_limited']]
        if sub:
            pr_by_T[T] = float(np.mean([r['projection_rate'] for r in sub]))
    pr_vals = [pr_by_T[T] for T in T_VALUES if T in pr_by_T]
    if len(pr_vals) >= 2 and all(pr_vals[i] <= pr_vals[i+1] for i in range(len(pr_vals)-1)):
        print("  MONOTONE PROJECTION INCREASE")
    else:
        print(f"  projection_rate not monotone in T (eps=0.1 values: "
              f"{[f'{v:.3f}' for v in pr_vals]})")

    # 3. Chain propagation: activation_step grows with T
    act_by_T = {}
    for T in T_VALUES:
        sub = [r for r in rows if r['T'] == T and r['epsilon'] == 0.1]
        if sub:
            acts = [r['last_coord_activation_step'] if r['last_coord_activation_step'] != -1
                    else N_MAX for r in sub]
            act_by_T[T] = float(np.median(acts))
    act_vals = [act_by_T[T] for T in T_VALUES if T in act_by_T]
    if len(act_vals) >= 2 and all(act_vals[i] <= act_vals[i+1] for i in range(len(act_vals)-1)):
        print("  CHAIN PROPAGATION OBSERVED")
    else:
        print(f"  activation_step not monotone in T (eps=0.1 values: "
              f"{[f'{int(v):,}' for v in act_vals]})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rows = load()
    print(f"Loaded {len(rows)} unique rows from {RESULTS_FILE}")
    T_seen = sorted(set(r['T'] for r in rows))
    print(f"T values present: {T_seen}")

    rng   = np.random.default_rng(42)
    table = summary_table(rows)

    print("\n--- Slope fits ---")
    slopes = plot_slope_vs_T(rows, rng)

    print("\n--- Projection activity ---")
    plot_projection_vs_T(rows)

    print("\n--- Activation steps ---")
    plot_activation_vs_T(rows)

    check_signatures(rows, slopes, table)
