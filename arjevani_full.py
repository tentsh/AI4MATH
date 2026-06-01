"""
Arjevani zero-chain with radial quadratic term and Arjevani rescaling.
Builds on arjevani_rotated.py.
"""

import numpy as np
from arjevani_rotated import make_U, hat_F, grad_hat_F

ETA_0 = 0.2
ELL_1 = 152        # Lipschitz constant from Arjevani et al. 2023, Section 3


# ---------------------------------------------------------------------------
# hat_F_full : hat_F + radial quadratic
# ---------------------------------------------------------------------------

def hat_F_full(x: np.ndarray, U: np.ndarray) -> float:
    """hat_F(x) + (eta_0 / 2) * ||x||^2"""
    return hat_F(x, U) + 0.5 * ETA_0 * float(np.dot(x, x))


def grad_hat_F_full(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """grad hat_F(x) + eta_0 * x"""
    return grad_hat_F(x, U) + ETA_0 * x


# ---------------------------------------------------------------------------
# Rescaling  (Section 3)
# ---------------------------------------------------------------------------

def make_lambda(epsilon: float, L_target: float = 1.0) -> float:
    """lambda = 4 * ell_1 * epsilon / L_target"""
    return 4.0 * ELL_1 * epsilon / L_target


def F_scaled(x: np.ndarray, U: np.ndarray,
             lam: float, L_target: float = 1.0) -> float:
    """(L_target * lambda^2 / ell_1) * hat_F_full(x / lambda)"""
    return (L_target * lam**2 / ELL_1) * hat_F_full(x / lam, U)


def grad_F_scaled(x: np.ndarray, U: np.ndarray,
                  lam: float, L_target: float = 1.0) -> np.ndarray:
    """
    d/dx F_scaled(x) = (L_target * lambda / ell_1) * grad_hat_F_full(x / lambda)

    Derivation: d/dx [(c * lambda^2) * g(x/lambda)]
                = (c * lambda^2) * (1/lambda) * grad_g(x/lambda)
                = (c * lambda) * grad_g(x/lambda),  c = L_target/ell_1.
    """
    return (L_target * lam / ELL_1) * grad_hat_F_full(x / lam, U)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_ball(n: int, d: int, R: float, rng: np.random.Generator) -> np.ndarray:
    """n points uniformly distributed in the ball of radius R in R^d."""
    dirs = rng.standard_normal((n, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.random(n) ** (1.0 / d)          # correct volume weighting
    return dirs * (R * radii)[:, None]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    T, d      = 10, 20
    epsilon   = 0.2
    L_target  = 1.0
    D_X       = 1.0           # diameter of the domain  =>  ball radius = D_X / 2

    U   = make_U(d, T, seed=0)
    lam = make_lambda(epsilon, L_target)

    print(f"lambda   = {lam:.6f}")
    print(f"L_target = {L_target}")
    print(f"ell_1    = {ELL_1}")
    print(f"Rescaling factor (function)  L*lam^2/ell_1 = {L_target * lam**2 / ELL_1:.6f}")
    print(f"Rescaling factor (gradient)  L*lam/ell_1   = {L_target * lam / ELL_1:.6f}")

    # 200 uniform samples from the ball
    rng    = np.random.default_rng(42)
    pts    = sample_ball(200, d, D_X / 2, rng)

    f_vals  = np.array([F_scaled(p, U, lam, L_target) for p in pts])
    g_norms = np.array([np.linalg.norm(grad_F_scaled(p, U, lam, L_target)) for p in pts])

    print(f"\n||grad_F_scaled||  min={g_norms.min():.4f}  "
          f"median={np.median(g_norms):.4f}  max={g_norms.max():.4f}")
    print(f"F_scaled           min={f_vals.min():.4f}  "
          f"median={np.median(f_vals):.4f}  max={f_vals.max():.4f}")

    # Sanity check
    g_med = float(np.median(g_norms))
    if g_med < 1e-6:
        print("\nWARNING: gradient norms suspiciously small (< 1e-6) — check rescaling.")
    elif g_med > 1e6:
        print("\nWARNING: gradient norms suspiciously large (> 1e6) — check rescaling.")
    else:
        print("\nSanity check OK: median gradient norm in reasonable range.")

    # Finite-difference gradient check on a single point
    print("\n--- Finite-difference check on one point ---")
    x0    = pts[0]
    g_ana = grad_F_scaled(x0, U, lam, L_target)
    h     = 1e-6
    g_fd  = np.zeros(d)
    for k in range(d):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        g_fd[k] = (F_scaled(xp, U, lam, L_target) - F_scaled(xm, U, lam, L_target)) / (2.0 * h)

    max_err = np.max(np.abs(g_ana - g_fd))
    print(f"Max |analytic - FD| = {max_err:.2e}  (threshold: 1e-4)")
    assert max_err < 1e-4, f"Gradient check FAILED: max error {max_err:.2e}"
    print("Gradient check PASSED.")
