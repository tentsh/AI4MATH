"""
Arjevani zero-chain with random rotation and soft projection.
Builds on arjevani_base.py.
"""

import numpy as np
from arjevani_base import F_T, grad_F_T


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------

def make_U(d: int, T: int, seed: int = 0) -> np.ndarray:
    """
    U in R^{d x T} with orthonormal columns.
    Draw a d x T Gaussian matrix, return the Q factor (reduced QR).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, T)))
    return Q          # shape (d, T), columns are orthonormal


# ---------------------------------------------------------------------------
# Soft projection and its Jacobian
# ---------------------------------------------------------------------------

def rho(z: np.ndarray) -> np.ndarray:
    """Coordinatewise: z_i / max(1, |z_i|).  Clips each coordinate to [-1, 1]."""
    return z / np.maximum(1.0, np.abs(z))


def jac_rho_diag(z: np.ndarray) -> np.ndarray:
    """
    Diagonal of Jrho at z (subgradient choice from the paper):
      1  if |z_i| <= 1
      0  if |z_i| >  1
    """
    return (np.abs(z) <= 1.0).astype(float)


# ---------------------------------------------------------------------------
# Composed function hat_F : R^d -> R
# ---------------------------------------------------------------------------

def hat_F(x: np.ndarray, U: np.ndarray) -> float:
    """hat_F(x) = F_T(rho(U.T @ x))"""
    return F_T(rho(U.T @ x))


def grad_hat_F(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Chain rule:  grad hat_F(x) = U @ (Jrho.T @ grad_F_T(rho_z)),  z = U.T @ x.
    Jrho is diagonal so Jrho.T = Jrho; the matrix-vector product reduces to
    element-wise multiplication by the diagonal.
    """
    z     = U.T @ x
    rho_z = rho(z)
    d_rho = jac_rho_diag(z)                 # diagonal of Jrho
    g_ft  = grad_F_T(rho_z)                 # gradient in R^T
    return U @ (d_rho * g_ft)               # back to R^d


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    T, d = 10, 20
    U = make_U(d, T, seed=0)

    # Verify U has orthonormal columns
    err_orth = np.max(np.abs(U.T @ U - np.eye(T)))
    print(f"U shape: {U.shape},  max |U^T U - I| = {err_orth:.2e}")

    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, d)

    fval  = hat_F(x, U)
    gval  = grad_hat_F(x, U)

    print(f"\nhat_F(x)           = {fval:.10f}")
    print(f"||grad_hat_F(x)||  = {np.linalg.norm(gval):.10f}")
    print(f"grad_hat_F(x):\n{gval}")

    # Finite-difference check (central differences, h=1e-6)
    h    = 1e-6
    g_fd = np.zeros(d)
    for k in range(d):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        g_fd[k] = (hat_F(xp, U) - hat_F(xm, U)) / (2.0 * h)

    max_err = np.max(np.abs(gval - g_fd))
    print(f"\nMax |analytic - FD| = {max_err:.2e}  (threshold: 1e-4)")
    assert max_err < 1e-4, f"Gradient check FAILED: max error {max_err:.2e}"
    print("Gradient check PASSED.")
