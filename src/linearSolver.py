# linear_solver.py
import copy
import numpy as np


def tdma(
        a_lower: np.ndarray,
        a_diag: np.ndarray,
        a_upper: np.ndarray,
        b: np.ndarray,
        ) -> np.ndarray:
    """
    Simple TDMA (Thomas algorithm) solver for tridiagonal systems.

    """

    N = len(b)

    # Copy matrix coefficients so that they're not overwritten
    c = a_upper.astype(float).copy()
    d = b.astype(float).copy()
    a = a_lower.astype(float).copy()
    b_diag = a_diag.astype(float).copy()

    # Forward elimination
    for i in range(1, N):
        m = a[i] / b_diag[i - 1]      # multiplier
        b_diag[i] = b_diag[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back-substitution
    x = np.zeros(N, dtype=float)
    x[-1] = d[-1] / b_diag[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b_diag[i]

    return x



if __name__ == "__main__":
    import numpy as np

    print("Running basic TDMA self-tests…")

    # ------------------------------------------------------------
    # Test 1: 1×1 system
    # ------------------------------------------------------------
    a_lower = np.array([0.0])
    a_diag  = np.array([2.0])
    a_upper = np.array([0.0])
    b       = np.array([8.0])
    x = tdma(a_lower, a_diag, a_upper, b)
    print("Test 1 (1x1):", x, " expected [4]")

    # ------------------------------------------------------------
    # Test 2: simple 2×2 system
    #
    #   [ 2  1 ] [x0] = [5]
    #   [ 3  4 ] [x1]   [6]
    # ------------------------------------------------------------
    a_lower = np.array([0.0, 3.0])
    a_diag  = np.array([2.0, 4.0])
    a_upper = np.array([1.0, 0.0])
    b       = np.array([5.0, 6.0])
    x = tdma(a_lower, a_diag, a_upper, b)
    # Expected:
    #   x0 = 0.8
    #   x1 = 1.4
    print("Test 2 (2x2):", x, " expected [0.8, 1.4]")

    # ------------------------------------------------------------
    # Test 3: random 5×5 system
    # ------------------------------------------------------------
    rng = np.random.default_rng(42)

    N = 5
    a_diag  = rng.uniform(1.0, 3.0, size=N)
    a_lower = rng.uniform(-1.0, 1.0, size=N)
    a_upper = rng.uniform(-1.0, 1.0, size=N)
    a_lower[0] = 0.0
    a_upper[-1] = 0.0

    b = rng.uniform(-2.0, 2.0, size=N)

    # Full matrix for comparison
    A = np.zeros((N, N))
    np.fill_diagonal(A, a_diag)
    for i in range(1, N):
        A[i, i - 1] = a_lower[i]
    for i in range(N - 1):
        A[i, i + 1] = a_upper[i]

    x_expected = np.linalg.solve(A, b)
    x = tdma(a_lower, a_diag, a_upper, b)

    print("Test 3 (random 5x5):")
    print("TDMA     :", x)
    print("Expected :", x_expected)
    print("Close?   :", np.allclose(x, x_expected))