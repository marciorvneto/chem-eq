import numpy as np
import pprint
from numerical import newton_raphson
from thermo import gamma_debye_huckel
import matplotlib.pyplot as plt
from loader import load_thermo_database
from helper import create_feed, to_mole_fractions, to_species_dict


db = load_thermo_database("thermo_db.csv")

nF = create_feed(db, {"H2O(l)": 55.508, "Na+(aq)": 0.5, "Cl-(aq)": 0.5, "CO2(aq)": 0.1})

T = 300


def solve_equilibrium(nF, T, tol_gamma=1e-6):
    atom_matrix = db["atom_matrix"]
    N = len(db["species"])
    C = atom_matrix.shape[1]
    n = nF
    lamb = np.ones(C)
    gamma = np.ones(nF.shape)
    nu = 1.0
    nT = 1.0
    total_atoms = nF.T @ atom_matrix

    err = 1000
    for i in range(5):
        n, nT, lamb, nu = newton_raphson(
            to_mole_fractions(n), nT, lamb, nu, T, gamma, total_atoms, db
        )
        gamma_new = gamma_debye_huckel(n, nT, T, db)
        err = np.linalg.norm(gamma - gamma_new)
        if err < tol_gamma:
            print(f"Outer loop converged in {i+1} iterations, with an error of {err}")
            break
        gamma = gamma_new

    return n, nT


n_eq, nT_eq = solve_equilibrium(nF, T)
output_sepecies = to_species_dict(db, n_eq, threshold=1e-6)
print("##### Feed #####")
pprint.pprint(to_species_dict(db, nF, threshold=1e-6))
print("##### Result #####")
pprint.pprint(output_sepecies)

# ########################################
# #
# #  Single case
# #
# ########################################
#
# n_eq, nT_eq, lamb_eq, nu_eq = newton_raphson(
#     n0, nT0, lamb0, nu0, T, gamma0, total_atoms
# )
#
# ########################################
# #
# #  Sweep
# #
# ########################################
#
# T_range = np.linspace(300, 1200, 50)
# mole_fractions = []
#
# # Start with your converged 300K roots to kick things off
# n_guess = n_eq.copy()
# gamma_guess = np.ones(n_eq.shape)
# nT_guess = nT_eq
# lamb_guess = lamb_eq.copy()
# nu_guess = nu_eq
#
# # 3. Run the continuation loop
# for T_step in T_range:
#     # Notice we feed the previous guess into the next step
#     n_guess, nT_guess, lamb_guess, nu_guess = newton_raphson(
#         n_guess,
#         nT_guess,
#         lamb_guess,
#         nu_guess,
#         T_step,
#         gamma_guess,
#         total_atoms,
#         tol=1e-8,
#         max_iter=100,
#     )
#
#     # Store the mole fractions (y_i = n_i / n_T)
#     mole_fractions.append(n_guess / nT_guess)
#
# mole_fractions = np.array(mole_fractions)
#
# # 4. Plot the magic
# plt.figure(figsize=(10, 6))
#
# for i, sp in enumerate(species):
#     # Only plot species that actually exist in meaningful amounts
#     if np.max(mole_fractions[:, i]) > 1e-4:
#         plt.plot(T_range, mole_fractions[:, i], label=sp, linewidth=2.5)
#
# plt.xlabel("Temperature (K)", fontsize=12)
# plt.ylabel("Mole Fraction ($y_i$)", fontsize=12)
# plt.title("Syngas System Equilibrium Composition vs. Temperature", fontsize=14)
# plt.legend(loc="best", fontsize=10)
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()
