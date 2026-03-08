import numpy as np

_R = 8.314

#############################################
#
#   Lagrangian gradients and jacobian
#
#############################################


def potential_RT(n, nT, T, gamma, db):
    h_f_std = db["dHf_std"]
    s_f_std = db["dSf_std"]
    g_f_T = h_f_std - T * s_f_std
    return g_f_T / (_R * T) + np.log(n) - np.log(nT) + np.log(gamma)


def gibbs_RT(n, nT, T, gamma, db):
    mu = potential_RT(n, nT, T, gamma, db)
    return n.dot(mu)


def lagrangian(n, nT, lamb, nu, T, gamma, total_atoms, db):
    atom_matrix = db["atom_matrix"]
    atom_balances = n.T @ atom_matrix - total_atoms
    total_moles_balance = np.sum(n) - nT
    lamb_term = lamb.T.dot(atom_balances.T)
    nu_term = nu * total_moles_balance
    return gibbs_RT(n, nT, T, gamma, db) + lamb_term + nu_term


def grad_lagrangian(n, nT, lamb, nu, T, gamma, total_atoms, db):
    return np.concatenate(
        [
            d_lagrangian_n(n, nT, lamb, nu, T, gamma, total_atoms).T,
            np.array([d_lagrangian_nT(n, nT, lamb, nu, T, gamma, total_atoms)]),
            d_lagrangian_lamb(n, nT, lamb, nu, T, gamma, total_atoms).T,
            np.array([d_lagrangian_nu(n, nT, lamb, nu, T, gamma, total_atoms)]),
        ]
    )


def d_lagrangian_n(n, nT, lamb, nu, T, gamma, total_atoms, db):
    atom_matrix = db["atom_matrix"]
    return (
        potential_RT(n, nT, T, gamma, db).T
        + n.T.dot(np.diag(1 / n))
        + lamb.T.dot(atom_matrix.T)
        + nu * np.ones(len(n))
    )


def d_lagrangian_nT(n, nT, lamb, nu, T, gamma, total_atoms, db):
    return -np.sum(n) / nT - nu


def d_lagrangian_lamb(n, nT, lamb, nu, T, gamma, total_atoms, db):
    atom_matrix = db["atom_matrix"]
    return n.T.dot(atom_matrix) - total_atoms.T


def d_lagrangian_nu(n, nT, lamb, nu, T, gamma, total_atoms, db):
    return np.sum(n) - nT


def lagrangian_jac(n, nT, lamb, nu, T, gamma, total_atoms, db):
    atom_matrix = db["atom_matrix"]
    N = len(n)
    C = len(lamb)

    j_n_n = np.diag(1 / n)
    j_n_nT = np.array([[-np.ones((N, 1)) / nT]])
    j_n_lamb = atom_matrix
    j_n_nu = np.ones((N, 1))

    j_nT_n = -1 / nT * np.ones((1, N))
    j_nT_nT = np.array([[np.sum(n) / (nT * nT)]])
    j_nT_lamb = np.zeros((1, C))
    j_nT_nu = np.array([[-1]])

    j_lamb_n = atom_matrix.T
    j_lamb_nT = np.zeros((C, 1))
    j_lamb_lamb = np.zeros((C, C))
    j_lamb_nu = np.zeros((C, 1))

    j_nu_n = np.ones((1, N))
    j_nu_nT = np.array([[-1]])
    j_nu_lamb = np.zeros((1, C))
    j_nu_nu = np.array([[0]])

    J = np.block(
        [
            [j_n_n, j_n_nT, j_n_lamb, j_n_nu],
            [j_nT_n, j_nT_nT, j_nT_lamb, j_nT_nu],
            [j_lamb_n, j_lamb_nT, j_lamb_lamb, j_lamb_nu],
            [j_nu_n, j_nu_nT, j_nu_lamb, j_nu_nu],
        ]
    )

    return J[0, 0, :, :]


#############################################
#
#   Activity models
#
#############################################


def gamma_debye_huckel(n, nT, T, db):
    """Extended Debye-Hückel for aqueous solutes, Raoult's law for solvent."""
    species = db["species"]
    molar_mass = db["molar_mass"]
    Z = db["Z"]
    idx_w = species.index("H2O(l)")

    # Molality calculation
    moles_water = max(n[idx_w], 1e-12)
    kg_water = moles_water * molar_mass[idx_w]
    molality = n / kg_water
    molality[idx_w] = 0.0  # Solvent doesn't contribute to its own molality

    # Ionic Strength
    I = 0.5 * np.sum(molality * (Z**2))

    A = 1.172  # Approx for 298K
    B_a = 1.0

    ln_gamma = -A * (Z**2) * np.sqrt(I) / (1.0 + B_a * np.sqrt(I))
    gamma = np.exp(ln_gamma)
    gamma[idx_w] = 1.0  # Ideal solvent

    return gamma


#######################################
#
#   Numerical approximation tests
#
#######################################


def test_analytical_approximations():
    nF = np.array(
        [
            1e-12,  # H2
            1e-12,  # N2
            1e-12,  # O2
            3.0,  # H2O
            3.0,  # CO
            3.0,  # CO2
            3.0,  # CH4
            3.0,  # NH3
            1e-12,  # NO
            1e-12,  # NO2
        ]
    )
    n0 = np.array(
        [
            3,  # H2
            3,  # N2
            3,  # O2
            3.0,  # H2O
            3.0,  # CO
            3.0,  # CO2
            3.0,  # CH4
            3.0,  # NH3
            3,  # NO
            3,  # NO2
        ]
    )
    lamb0 = np.array(
        [
            1.0,  # C
            1.0,  # H
            1.0,  # O
            1.0,  # N
        ]
    )
    gamma0 = np.abs(np.random.random(n0.shape))
    nu0 = 1.0
    nT0 = 1.0
    T = 400
    total_atoms = nF.T @ atom_matrix
    N = len(n0)
    C = len(lamb0)

    dn = np.abs(np.random.randn(N)) * 1e-12
    dnT = np.abs(np.random.randn()) * 1e-12
    dlamb = np.abs(np.random.randn(C)) * 1e-12
    dnu = np.abs(np.random.randn()) * 1e-12

    #######################################
    #
    #   Gradient check
    #
    #######################################

    print("")
    print("##############################")
    print("#                            #")
    print("#      Gradient tests        #")
    print("#                            #")
    print("##############################")
    print("")

    L = lagrangian(n0, nT0, lamb0, nu0, T, gamma0, total_atoms)

    dL_num_n = lagrangian(n0 + dn, nT0, lamb0, nu0, T, gamma0, total_atoms) - L
    dL_ana_n = d_lagrangian_n(n0, nT0, lamb0, nu0, T, gamma0, total_atoms).dot(dn)
    print(f"dL_num_n = {dL_num_n}")
    print(f"dL_ana_n = {dL_ana_n}")
    error = np.linalg.norm(dL_num_n - dL_ana_n)
    print(f"Error: {error}")

    dL_num_nT = lagrangian(n0, nT0 + dnT, lamb0, nu0, T, gamma0, total_atoms) - L
    dL_ana_nT = d_lagrangian_nT(n0, nT0, lamb0, nu0, T, gamma0, total_atoms) * dnT
    print(f"dL_num_nT = {dL_num_nT}")
    print(f"dL_ana_nT = {dL_ana_nT}")
    error = np.linalg.norm(dL_num_nT - dL_ana_nT)
    print(f"Error: {error}")

    dL_num_lamb = lagrangian(n0, nT0, lamb0 + dlamb, nu0, T, gamma0, total_atoms) - L
    dL_ana_lamb = d_lagrangian_lamb(n0, nT0, lamb0, nu0, T, gamma0, total_atoms).dot(
        dlamb
    )
    print(f"dL_num_lamb = {dL_num_lamb}")
    print(f"dL_ana_lamb = {dL_ana_lamb}")
    error = np.linalg.norm(dL_num_lamb - dL_ana_lamb)
    print(f"Error: {error}")

    dL_num_nu = lagrangian(n0, nT0, lamb0, nu0 + dnu, T, gamma0, total_atoms) - L
    dL_ana_nu = d_lagrangian_nu(n0, nT0, lamb0, nu0, T, gamma0, total_atoms) * dnu
    print(f"dL_num_nu = {dL_num_nu}")
    print(f"dL_ana_nu = {dL_ana_nu}")
    error = np.linalg.norm(dL_num_nu - dL_ana_nu)
    print(f"Error: {error}")

    #######################################
    #
    #   Jacobian check
    #
    #######################################

    print("")
    print("##############################")
    print("#                            #")
    print("#      Jacobian tests        #")
    print("#                            #")
    print("##############################")
    print("")

    gL = grad_lagrangian(n0, nT0, lamb0, nu0, T, gamma0, total_atoms)
    jac = lagrangian_jac(n0, nT0, lamb0, nu0, T, gamma0, total_atoms)

    dgradL_num_n = (
        grad_lagrangian(n0 + dn, nT0, lamb0, nu0, T, gamma0, total_atoms) - gL
    )
    dvars = np.concatenate(
        [
            dn,
            0 * np.array([nT0]),
            0 * lamb0,
            0 * np.array([nu0]),
        ]
    )
    dgrad_ana = jac @ dvars
    print(f"dgradL_num_n = {dgradL_num_n[:N]}")
    print(f"dgradL_ana_n = {dgrad_ana[:N]}")
    error = np.linalg.norm(dgradL_num_n[:N] - dgrad_ana[:N])
    print(f"Error: {error}")

    dgradL_num_nT = (
        grad_lagrangian(n0, nT0 + dnT, lamb0, nu0, T, gamma0, total_atoms) - gL
    )
    dvars = np.concatenate(
        [
            0 * dn,
            np.array([dnT]),
            0 * lamb0,
            0 * np.array([nu0]),
        ]
    )
    dgrad_ana = jac @ dvars
    print(f"dgradL_num_nT = {dgradL_num_nT[N:N+1]}")
    print(f"dgradL_ana_nT = {dgrad_ana[N:N+1]}")
    error = np.linalg.norm(dgradL_num_nT[N : N + 1] - dgrad_ana[N : N + 1])
    print(f"Error: {error}")

    dgradL_num_lamb = (
        grad_lagrangian(n0, nT0, lamb0 + dlamb, nu0, T, gamma0, total_atoms) - gL
    )
    dvars = np.concatenate(
        [
            0 * dn,
            0 * np.array([dnT]),
            dlamb,
            0 * np.array([nu0]),
        ]
    )
    dgrad_ana = jac @ dvars
    print(f"dgradL_num_lamb = {dgradL_num_lamb[N+1:N+1+C]}")
    print(f"dgradL_ana_lamb = {dgrad_ana[N+1:N+1+C]}")
    error = np.linalg.norm(
        dgradL_num_lamb[N + 1 : N + 1 + C] - dgrad_ana[N + 1 : N + 1 + C]
    )
    print(f"Error: {error}")

    dgradL_num_nu = (
        grad_lagrangian(n0, nT0, lamb0, nu0 + dnu, T, gamma0, total_atoms) - gL
    )
    dvars = np.concatenate(
        [
            0 * dn,
            0 * np.array([dnT]),
            0 * lamb0,
            np.array([dnu]),
        ]
    )
    dgrad_ana = jac @ dvars
    print(f"dgradL_num_nu = {dgradL_num_nu[N+1+C:]}")
    print(f"dgradL_ana_nu = {dgrad_ana[N+1+C:]}")
    error = np.linalg.norm(dgradL_num_nu[N + 1 + C :] - dgrad_ana[N + 1 + C :])
    print(f"Error: {error}")
