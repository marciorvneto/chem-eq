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
