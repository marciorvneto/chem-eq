import numpy as np


def create_feed(db, feed_dict):
    species = db["species"]
    N = len(species)
    nF = np.ones(N) * 1e-12
    for sp, moles in feed_dict.items():
        if sp in species:
            nF[species.index(sp)] = moles
        else:
            print(f"Warning: {sp} not found in database!")
    return nF


def to_mole_fractions(n):
    return n / np.sum(n)


def to_species_dict(db, n, threshold=1e-20):
    species = db["species"]
    N = len(species)
    species_dict = {}
    mole_fractions = to_mole_fractions(n)
    for i, sp in enumerate(species):
        if mole_fractions[species.index(sp)] <= threshold:
            continue
        species_dict[sp] = n[species.index(sp)]
    return species_dict
