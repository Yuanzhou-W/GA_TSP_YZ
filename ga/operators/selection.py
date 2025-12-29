# ga/operators/selection.py

import numpy as np


# --------------------------------------------------
# Roulette Wheel Selection
# --------------------------------------------------

def roulette_wheel_selection(fitness, num_selected):
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

def select(fitness, method="roulette", num_selected=None):
    """
    Selection interface (RETURN INDICES ONLY)

    Parameters
    ----------
    fitness : np.ndarray
    method : str
        'roulette' or 'sus'
    num_selected : int
    """
    pop_size = len(fitness)
    if num_selected is None:
        num_selected = pop_size

    if method == "roulette":
        return roulette_wheel_selection(fitness, num_selected)
    elif method == "sus":
        return stochastic_universal_sampling(fitness, num_selected)
    else:
        raise ValueError(f"Unknown selection method: {method}")
