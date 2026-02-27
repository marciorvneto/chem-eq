import pprint
from numerical import solve_equilibrium
from loader import load_thermo_database
from helper import create_feed, to_species_dict

T = 300
db = load_thermo_database("thermo_db.csv")
feed = create_feed(
    db, {"H2O(l)": 55.508, "Na+(aq)": 0.5, "Cl-(aq)": 0.5, "CO2(aq)": 0.1}
)

n_eq, nT_eq = solve_equilibrium(db, feed, T)
output_sepecies = to_species_dict(db, n_eq, threshold=1e-6)
print("##### Feed #####")
pprint.pprint(to_species_dict(db, feed, threshold=1e-6))
print("##### Result #####")
pprint.pprint(output_sepecies)
