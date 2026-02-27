import numpy as np
import pprint
from numerical import newton_raphson, solve_equilibrium
from thermo import gamma_debye_huckel
import matplotlib.pyplot as plt
from loader import load_thermo_database
from helper import create_feed, to_mole_fractions, to_species_dict

T = 300
db = load_thermo_database("thermo_db.csv")
nF = create_feed(db, {"H2O(l)": 55.508, "Na+(aq)": 0.5, "Cl-(aq)": 0.5, "CO2(aq)": 0.1})

n_eq, nT_eq = solve_equilibrium(nF, T)
output_sepecies = to_species_dict(db, n_eq, threshold=1e-6)
print("##### Feed #####")
pprint.pprint(to_species_dict(db, nF, threshold=1e-6))
print("##### Result #####")
pprint.pprint(output_sepecies)
