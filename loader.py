import numpy as np
import csv


def load_thermo_database(filepath="thermo_db.csv"):
    """Reads the database and returns the thermodynamic arrays."""

    species = []
    states = []
    h_f_std_list = []
    g_f_std_list = []
    molar_mass_list = []
    atom_matrix_list = []

    with open(filepath, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip the header row

        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue  # Skip empty lines or comments

            # Clean up whitespace
            row = [col.strip() for col in row]

            species.append(row[0])
            states.append(row[1])
            h_f_std_list.append(float(row[2]))
            g_f_std_list.append(float(row[3]))
            molar_mass_list.append(float(row[4]))

            # Elements + Charge (Columns 5 through end)
            atom_matrix_list.append([float(x) for x in row[5:]])

    # Convert to NumPy arrays for the math engine
    h_f_std = np.array(h_f_std_list)
    g_f_std = np.array(g_f_std_list)
    molar_mass = np.array(molar_mass_list)
    atom_matrix = np.array(atom_matrix_list)

    # Pre-calculate entropy of formation
    s_f_std = (h_f_std - g_f_std) / 298.15

    # Extract charges (the last column of the atom matrix)
    Z = atom_matrix[:, -1]

    # Universal gas constant
    _R = 8.314462618

    return {
        "species": species,
        "states": states,
        "h_f_std": h_f_std,
        "g_f_std": g_f_std,
        "s_f_std": s_f_std,
        "molar_mass": molar_mass,
        "atom_matrix": atom_matrix,
        "Z": Z,
        "_R": _R,
    }


# --- Quick Test ---
if __name__ == "__main__":
    db = load_thermo_database("thermo_db.csv")
    print(f"Loaded {len(db['species'])} species successfully.")
    print(f"Elements tracked: {db['atom_matrix'].shape[1] - 1} + 1 Charge column")
    print(
        f"H2O(l) molar mass: {db['molar_mass'][db['species'].index('H2O(l)')]} kg/mol"
    )
