# ga/operators/selection.py

import numpy as np


# --------------------------------------------------
# Roulette Wheel Selection
# --------------------------------------------------

def roulette_wheel_selection(fitness, num_selected):
    """
    Roulette Wheel Selection based on fitness.

    Parameters
    ----------
    fitness : np.ndarray
        Fitness values (higher is better)
    num_selected : int
        Number of individuals to select
    """
    fitness = np.asarray(fitness)

    # 防止负值 / 全零
    fitness = fitness - fitness.min() + 1e-12
    probs = fitness / fitness.sum()

    cum_probs = np.cumsum(probs)

    r = np.random.rand(num_selected)
    indices = np.searchsorted(cum_probs, r)

    return indices


# --------------------------------------------------
# Stochastic Universal Sampling (SUS)
# --------------------------------------------------

def stochastic_universal_sampling(fitness, num_selected):
    """
    Stochastic Universal Sampling (SUS)
    """
    fitness = np.asarray(fitness)

    fitness = fitness - fitness.min() + 1e-12
    probs = fitness / fitness.sum()

    cum_probs = np.cumsum(probs)

    step = 1.0 / num_selected
    start = np.random.rand() * step
    pointers = start + step * np.arange(num_selected)

    indices = np.searchsorted(cum_probs, pointers)

    return indices


# --------------------------------------------------
# Selection Dispatcher
# --------------------------------------------------

def select(population, fitness, method="roulette", num_selected=None):
    """
    Selection interface.

    Parameters
    ----------
    population : np.ndarray
    fitness : np.ndarray
    method : str
        'roulette' or 'sus'
    num_selected : int or None
        Number of individuals to select
    """
    pop_size = len(population)

    if num_selected is None:
        num_selected = pop_size

    if method == "roulette":
        indices = roulette_wheel_selection(fitness, num_selected)
    elif method == "sus":
        indices = stochastic_universal_sampling(fitness, num_selected)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return population[indices]
