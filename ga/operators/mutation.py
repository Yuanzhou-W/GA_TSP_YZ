# ga/operators/mutation.py

import numpy as np


# --------------------------------------------------
# Swap Mutation
# --------------------------------------------------

def swap_mutation(individual):
    """
    Swap mutation for TSP.
    """
    ind = individual.copy()
    i, j = np.random.choice(len(ind), 2, replace=False)
    ind[i], ind[j] = ind[j], ind[i]
    return ind


# --------------------------------------------------
# Inversion Mutation
# --------------------------------------------------

def inversion_mutation(individual):
    """
    Inversion mutation for TSP.
    """
    ind = individual.copy()
    i, j = sorted(np.random.choice(len(ind), 2, replace=False))
    ind[i:j] = ind[i:j][::-1]
    return ind


# --------------------------------------------------
# Dispatcher
# --------------------------------------------------

def mutate(individual, pm, method="swap"):
    """
    Mutation dispatcher.

    Parameters
    ----------
    individual : np.ndarray
        TSP permutation
    pm : float
        Mutation probability
    method : str
        'swap' or 'inversion'
    """
    if np.random.rand() >= pm:
        return individual.copy()

    if method == "swap":
        return swap_mutation(individual)
    elif method == "inversion":
        return inversion_mutation(individual)
    else:
        raise ValueError(f"Unknown mutation method: {method}")
