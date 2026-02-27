# ChemEQ

A simple, robust chemical equilibrium solver written in Python.

This library solves arbitrary chemical equilibrium scenarios by directly minimizing the total Gibbs free energy of the system.

**Note**: This is an experimental project. We currently only support the Extended Debye-Hückel activity model for aqueous species, and assume ideal gas behavior. The thermodynamic species database is a work in progress. Feel free to contribute!

**Interested in the theory?**: If you're interested in the theory, I wrote a paper on Gibbs minimization a while ago. Please feel free to refer to it: https://www.sciencedirect.com/science/article/abs/pii/S037838121730211X

## Features

- **Direct Gibbs Minimization**: Automatically calculates phase splits and reaction extents by finding the minimum energy state. No pre-defined stoichiometric equations or $K$-values required.

- **Non-Ideal Aqueous Thermodynamics**: Implements an outer Picard iteration (successive substitution) loop to resolve activity coefficients ($\gamma$) via the Extended Debye-Hückel model, capturing ionic strength and salting effects.

- **Robust NLP Solver**: Uses a custom Damped Newton-Raphson approach (similar to the RAND method) to strictly satisfy the KKT conditions.

- **Dynamic Database**: Thermodynamic data is decoupled from the math engine. The extensible plaintext database (`thermo_db.txt`) allows you to add new species simply by adding a row with standard formation properties.

## Installation

Clone the repository and install the single dependency (`numpy`):

```bash
git clone git@github.com:marciorvneto/chem-eq.git
cd chem-eq
pip install numpy
```

## Quick Start

You can define a feed vector and pass it to the solver. The engine will automatically satisfy mass balances and electroneutrality while finding the equilibrium phase distribution.

```python
import pprint
from numerical import solve_equilibrium
from loader import load_thermo_database
from helper import create_feed, to_species_dict

# Load the components database
db = load_thermo_database("thermo_db.csv")

# Define system temperature (K)
T = 300

# Define your feed (e.g., dissolving 0.1 moles of CO2 in 1 kg of salty water)
feed = create_feed({
    "H2O(l)": 55.508,
    "Na+(aq)": 0.5,
    "Cl-(aq)": 0.5,
    "CO2(aq)": 0.1
})

# Run the solver
n_eq, nT_eq = solve_equilibrium(db, feed, T)
```

**Output:**

```text
Converged in 47 iterations! Max KKT error: 3.64e-10
Outer loop converged in 3 iterations.

##### Result #####
{'CO2(aq)': 0.003119,
 'CO2(g)': 0.096436,
 'Cl-(aq)': 0.500000,
 'H+(aq)': 0.000443,
 'H2O(g)': 1.885430,
 'H2O(l)': 53.62212,
 'HCO3-(aq)': 0.000443,
 'Na+(aq)': 0.500000}

```

_Notice how the solver naturally discovers the VLE phase split (stripping CO2 into the gas phase) and the acid-base dissociation (generating H+ and HCO3-) simultaneously._

## How it Works

ChemEQ treats chemical equilibrium as a constrained non-linear optimization problem. It defines a Lagrangian function combining the total Gibbs free energy, elemental mass balances (via Lagrange multipliers, $\lambda$), and total mole constraints ($\nu$).

Instead of gradient descent, it analytically constructs the Jacobian (Hessian of the Lagrangian) and uses a damped Newton-Raphson step to find the roots of the KKT conditions. To handle the highly non-linear nature of activity models, the inner Newton solver assumes a constant $\gamma$ field, which is updated by an outer fixed-point iteration loop.

## Contributing

Contributions are highly welcome! Here are a few areas that need work:

- **Database Expansion**: Adding more species, ions, and solids to `thermo_db.txt`.
- **Solid Precipitation**: Expanding the solver logic to handle the appearance and disappearance of pure solid phases (checking saturation indices).
- **Advanced Activity Models**: Implementing Pitzer equations for high-salinity brines or Peng-Robinson for real gases.

```

```
