import numpy as np
from thermo import (
    d_lagrangian_n,
    d_lagrangian_nT,
    d_lagrangian_lamb,
    d_lagrangian_nu,
    lagrangian_jac,
)


def newton_raphson(
    n0, nT0, lamb0, nu0, T, gamma, total_atoms, db, tol=1e-8, max_iter=100
):
    n = n0.copy()
    nT = nT0
    lamb = lamb0.copy()
    nu = nu0

    N = len(n)
    C = len(lamb)

    for i in range(max_iter):
        # 1. Assemble the KKT gradient vector F
        F_n = d_lagrangian_n(n, nT, lamb, nu, T, gamma, total_atoms, db)
        F_nT = d_lagrangian_nT(n, nT, lamb, nu, T, gamma, total_atoms, db)
        F_lamb = d_lagrangian_lamb(n, nT, lamb, nu, T, gamma, total_atoms, db)
        F_nu = d_lagrangian_nu(n, nT, lamb, nu, T, gamma, total_atoms, db)

        # Flatten into a single 1D array
        F = np.concatenate(
            [
                F_n.flatten(),
                np.array([F_nT]).flatten(),
                F_lamb.flatten(),
                np.array([F_nu]).flatten(),
            ]
        )

        # Check for convergence (infinity norm)
        error = np.linalg.norm(F, np.inf)
        if error < tol:
            print(f"Converged in {i} iterations! Max KKT error: {error:.2e}")
            break

        # 2. Assemble the KKT Jacobian matrix J
        J = lagrangian_jac(n, nT, lamb, nu, T, gamma, total_atoms, db)

        # 3. Solve the linear system: J * dx = -F
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("Fatal: Singular Jacobian encountered.")
            break

        # Unpack the step vector
        dn = dx[:N]
        dnT = dx[N]
        dlamb = dx[N + 1 : N + 1 + C]
        dnu = dx[-1]

        # 4. Fraction of Maximum Step (Damping)
        # Find the largest alpha in (0, 1] that keeps all n > 0
        alpha = 1.0
        tau = 0.99  # Safety factor so we don't land exactly on 0

        for j in range(N):
            if dn[j] < 0:
                max_step = -n[j] / dn[j]
                alpha = min(alpha, tau * max_step)

        if dnT < 0:
            alpha = min(alpha, tau * (-nT / dnT))

        # 5. Apply the damped step
        n += alpha * dn
        nT += alpha * dnT
        lamb += alpha * dlamb
        nu += alpha * dnu

        # print(f"Iter {i:3d} | Error: {error:.2e} | Step size (alpha): {alpha:.4f}")

    else:
        print("Failed to converge within maximum iterations.")

    return n, nT, lamb, nu
