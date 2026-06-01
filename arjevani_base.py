"""
Base zero-chain from Arjevani et al. 2023 (arXiv:1912.02365), Section 3.

F_T : R^T -> R,  Eq. 15-16 in the paper (0-indexed throughout).
"""

import numpy as np
from scipy.stats import norm
import math

# ---------------------------------------------------------------------------
# Scalar bump functions
# ---------------------------------------------------------------------------

def Psi(u: float) -> float:
    """Eq. 16: 0 for u <= 1/2, else exp(1 - 1/(2u-1)^2)."""
    if u <= 0.5:
        return 0.0
    v = 2.0 * u - 1.0
    return math.exp(1.0 - 1.0 / v**2)


def dPsi(u: float) -> float:
    """d/du Psi(u).  Chain rule on exp(1 - 1/v^2), v=2u-1: Psi(u)*4/(2u-1)^3."""
    if u <= 0.5:
        return 0.0
    v = 2.0 * u - 1.0
    return Psi(u) * 4.0 / v**3


_SQRT_E   = math.sqrt(math.e)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def Phi(u: float) -> float:
    """Eq. 16: sqrt(e) * integral_{-inf}^{u} exp(-t^2/2) dt = sqrt(e)*sqrt(2pi)*norm.cdf(u)."""
    return _SQRT_E * _SQRT_2PI * norm.cdf(u)


def dPhi(u: float) -> float:
    """d/du Phi(u) = sqrt(e) * exp(-u^2/2).  Note: dPhi is an even function."""
    return _SQRT_E * math.exp(-u * u / 2.0)


# ---------------------------------------------------------------------------
# Zero-chain F_T
# ---------------------------------------------------------------------------

def F_T(x: np.ndarray) -> float:
    """
    F_T(x) = -Psi(1)*Phi(x[0])
             + sum_{i=1}^{T-1} [ Psi(-x[i-1])*Phi(-x[i]) - Psi(x[i-1])*Phi(x[i]) ]
    """
    T = len(x)
    val = -Psi(1.0) * Phi(float(x[0]))
    for i in range(1, T):
        val += Psi(-x[i-1]) * Phi(-x[i]) - Psi(x[i-1]) * Phi(x[i])
    return val


def grad_F_T(x: np.ndarray) -> np.ndarray:
    """
    Analytic gradient via chain rule.

    Each x[j] appears in at most two sum terms:
      - as x[i-1] in term i = j+1  (for 0 <= j <= T-2)
      - as x[i]   in term i = j    (for 1 <= j <= T-1)
    plus x[0] appears in the leading -Psi(1)*Phi(x[0]) term.
    """
    T = len(x)
    g = np.zeros(T)

    # ── Component 0 ───────────────────────────────────────────────────────
    # From leading term -Psi(1)*Phi(x[0]):
    g[0] -= Psi(1.0) * dPhi(float(x[0]))
    # From term i=1, x[0] enters as x[i-1]:
    #   d/dx[0] [ Psi(-x[0])*Phi(-x[1]) - Psi(x[0])*Phi(x[1]) ]
    #   = -dPsi(-x[0])*Phi(-x[1]) - dPsi(x[0])*Phi(x[1])
    if T > 1:
        g[0] -= dPsi(-x[0]) * Phi(-x[1]) + dPsi(x[0]) * Phi(x[1])

    # ── Middle components 1 .. T-2 ────────────────────────────────────────
    for j in range(1, T - 1):
        # x[j] as x[i-1] in term i=j+1:
        #   d/dx[j] [ Psi(-x[j])*Phi(-x[j+1]) - Psi(x[j])*Phi(x[j+1]) ]
        #   = -dPsi(-x[j])*Phi(-x[j+1]) - dPsi(x[j])*Phi(x[j+1])
        g[j] -= dPsi(-x[j]) * Phi(-x[j+1]) + dPsi(x[j]) * Phi(x[j+1])
        # x[j] as x[i] in term i=j:
        #   d/dx[j] [ Psi(-x[j-1])*Phi(-x[j]) - Psi(x[j-1])*Phi(x[j]) ]
        #   = -Psi(-x[j-1])*dPhi(-x[j]) - Psi(x[j-1])*dPhi(x[j])
        #   dPhi is even, so dPhi(-x[j]) = dPhi(x[j]):
        #   = -(Psi(-x[j-1]) + Psi(x[j-1])) * dPhi(x[j])
        g[j] -= (Psi(-x[j-1]) + Psi(x[j-1])) * dPhi(float(x[j]))

    # ── Last component T-1 ────────────────────────────────────────────────
    # x[T-1] as x[i] in term i=T-1:
    #   -(Psi(-x[T-2]) + Psi(x[T-2])) * dPhi(x[T-1])
    if T > 1:
        j = T - 1
        g[j] -= (Psi(-x[j-1]) + Psi(x[j-1])) * dPhi(float(x[j]))

    return g


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    T = 10
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, T)

    f_val = F_T(x)
    g_val = grad_F_T(x)

    print(f"T = {T}")
    print(f"x = {x}")
    print(f"\nF_T(x)      = {f_val:.10f}")
    print(f"grad_F_T(x) = {g_val}")

    # Finite-difference gradient check (central differences, h=1e-6)
    h = 1e-6
    g_fd = np.zeros(T)
    for k in range(T):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        g_fd[k] = (F_T(xp) - F_T(xm)) / (2.0 * h)

    max_err = np.max(np.abs(g_val - g_fd))
    print(f"\nFinite-difference gradient:\n{g_fd}")
    print(f"\nMax |analytic - FD| = {max_err:.2e}  (threshold: 1e-4)")
    assert max_err < 1e-4, f"Gradient check FAILED: max error {max_err:.2e}"
    print("Gradient check PASSED.")
