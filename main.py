import numpy as np
from thermo import g_f_std, _R, atom_matrix, species
from numerical import newton_raphson
import matplotlib.pyplot as plt
from loader import load_thermo_database
from helper import create_feed


db = load_thermo_database("thermo_db.csv")

feed = create_feed(
    db, {"H2O(l)": 55.508, "Na+(aq)": 0.5, "Cl-(aq)": 0.5, "CO2(aq)": 0.1}
)


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
        1e-6,  # H2
        1e-6,  # N2
        1e-6,  # O2
        1e-6,  # H2O
        1e-6,  # CO
        1e-6,  # CO2
        1e-6,  # CH4
        1e-6,  # NH3
        1e-6,  # NO
        1e-6,  # NO2
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
gamma0 = np.ones(n0.shape)
nu0 = 1.0
nT0 = 1.0
T = 400
total_atoms = nF.T @ atom_matrix
N = len(n0)
C = len(lamb0)


########################################
#
#  Single case
#
########################################

n_eq, nT_eq, lamb_eq, nu_eq = newton_raphson(
    n0, nT0, lamb0, nu0, T, gamma0, total_atoms
)

########################################
#
#  Sweep
#
########################################

T_range = np.linspace(300, 1200, 50)
mole_fractions = []

# Start with your converged 300K roots to kick things off
n_guess = n_eq.copy()
gamma_guess = np.ones(n_eq.shape)
nT_guess = nT_eq
lamb_guess = lamb_eq.copy()
nu_guess = nu_eq

# 3. Run the continuation loop
for T_step in T_range:
    # Notice we feed the previous guess into the next step
    n_guess, nT_guess, lamb_guess, nu_guess = newton_raphson(
        n_guess,
        nT_guess,
        lamb_guess,
        nu_guess,
        T_step,
        gamma_guess,
        total_atoms,
        tol=1e-8,
        max_iter=100,
    )

    # Store the mole fractions (y_i = n_i / n_T)
    mole_fractions.append(n_guess / nT_guess)

mole_fractions = np.array(mole_fractions)

# 4. Plot the magic
plt.figure(figsize=(10, 6))

for i, sp in enumerate(species):
    # Only plot species that actually exist in meaningful amounts
    if np.max(mole_fractions[:, i]) > 1e-4:
        plt.plot(T_range, mole_fractions[:, i], label=sp, linewidth=2.5)

plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel("Mole Fraction ($y_i$)", fontsize=12)
plt.title("Syngas System Equilibrium Composition vs. Temperature", fontsize=14)
plt.legend(loc="best", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
