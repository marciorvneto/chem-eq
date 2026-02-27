import numpy as np

# Species reference list (to keep track of the indices)
species = ["H2", "N2", "O2", "H2O", "CO", "CO2", "CH4", "NH3", "NO", "NO2"]

# Standard Enthalpy of formation at 298.15 K (J/mol)
h_f_std = np.array(
    [
        0.0,  # H2
        0.0,  # N2
        0.0,  # O2
        -241820.0,  # H2O
        -110530.0,  # CO
        -393510.0,  # CO2
        -74810.0,  # CH4
        -45900.0,  # NH3
        90290.0,  # NO
        33180.0,  # NO2
    ]
)

# Standard Gibbs free energy of formation at 298.15 K, 1 bar (in J/mol)
g_f_std = np.array(
    [
        0.0,  # H2
        0.0,  # N2
        0.0,  # O2
        -228570.0,  # H2O
        -137170.0,  # CO
        -394360.0,  # CO2
        -50720.0,  # CH4
        -16450.0,  # NH3
        86550.0,  # NO
        51310.0,  # NO2
    ]
)

# delta_G = delta_H - T * delta_S  =>  delta_S = (delta_H - delta_G) / T
s_f_std = (h_f_std - g_f_std) / 298.15

# Elemental abundance matrix (rows = species, columns = C, H, O, N)
# Arranged this way so you can easily transpose it for the mass balance constraint: A^T @ n = b
atom_matrix = np.array(
    [
        # C, H, O, N
        [0, 2, 0, 0],  # H2
        [0, 0, 0, 2],  # N2
        [0, 0, 2, 0],  # O2
        [0, 2, 1, 0],  # H2O
        [1, 0, 1, 0],  # CO
        [1, 0, 2, 0],  # CO2
        [1, 4, 0, 0],  # CH4
        [0, 3, 0, 1],  # NH3
        [0, 0, 1, 1],  # NO
        [0, 0, 2, 1],  # NO2
    ]
)

# Universal gas constant (J/(mol*K))
_R = 8.314462618


#############################################
#
#   Lagrangian grandients and jacobian
#
#############################################


def potential_RT(n, nT, T, gamma):
    g_f_T = h_f_std - T * s_f_std
    return g_f_T / (_R * T) + np.log(n) - np.log(nT) + np.log(gamma)


def gibbs_RT(n, nT, T, gamma):
    mu = potential_RT(n, nT, T, gamma)
    return n.dot(mu)


def lagrangian(n, nT, lamb, nu, T, gamma, total_atoms):
    atom_balances = n.T @ atom_matrix - total_atoms
    total_moles_balance = np.sum(n) - nT
    lamb_term = lamb.T.dot(atom_balances.T)
    nu_term = nu * total_moles_balance
    return gibbs_RT(n, nT, T, gamma) + lamb_term + nu_term


def grad_lagrangian(n, nT, lamb, nu, T, gamma, total_atoms):
    return np.concatenate(
        [
            d_lagrangian_n(n, nT, lamb, nu, T, gamma, total_atoms).T,
            np.array([d_lagrangian_nT(n, nT, lamb, nu, T, gamma, total_atoms)]),
            d_lagrangian_lamb(n, nT, lamb, nu, T, gamma, total_atoms).T,
            np.array([d_lagrangian_nu(n, nT, lamb, nu, T, gamma, total_atoms)]),
        ]
    )


def d_lagrangian_n(n, nT, lamb, nu, T, gamma, total_atoms):
    return (
        potential_RT(n, nT, T, gamma).T
        + n.T.dot(np.diag(1 / n))
        + lamb.T.dot(atom_matrix.T)
        + nu * np.ones(len(n))
    )


def d_lagrangian_nT(n, nT, lamb, nu, T, gamma, total_atoms):
    return -np.sum(n) / nT - nu


def d_lagrangian_lamb(n, nT, lamb, nu, T, gamma, total_atoms):
    return n.T.dot(atom_matrix) - total_atoms.T


def d_lagrangian_nu(n, nT, lamb, nu, T, gamma, total_atoms):
    return np.sum(n) - nT


def lagrangian_jac(n, nT, lamb, nu, T, gamma, total_atoms):
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


def test_analytical_approximations():

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
