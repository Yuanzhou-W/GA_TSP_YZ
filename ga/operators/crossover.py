# ga/operators/crossover.py

import numpy as np


# --------------------------------------------------
# Order Crossover (OX)
# --------------------------------------------------

def order_crossover(p1, p2):
    """
    Order Crossover (OX) for TSP.
    """
    size = len(p1)

    a, b = sorted(np.random.choice(size, 2, replace=False))

    c1 = [-1] * size
    c2 = [-1] * size

    # copy slice
    c1[a:b] = p1[a:b]
    c2[a:b] = p2[a:b]

    def fill(child, parent):
        pos = b
        for gene in parent:
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1

    fill(c1, p2)
    fill(c2, p1)

    return np.array(c1), np.array(c2)


# --------------------------------------------------
# PMX (optional, reserved)
# --------------------------------------------------

def pmx_crossover(p1, p2):
    """
    Partially Mapped Crossover (PMX)
    """
    size = len(p1)
    a, b = sorted(np.random.choice(size, 2, replace=False))

    c1 = p1.copy()
    c2 = p2.copy()

    mapping1 = {}
    mapping2 = {}

    for i in range(a, b):
        c1[i], c2[i] = c2[i], c1[i]
        mapping1[c1[i]] = c2[i]
        mapping2[c2[i]] = c1[i]

    def repair(child, mapping):
        for i in range(size):
            while child[i] in mapping:
                child[i] = mapping[child[i]]

    repair(c1, mapping1)
    repair(c2, mapping2)

    return c1, c2


# --------------------------------------------------
# Dispatcher
# --------------------------------------------------

def crossover(p1, p2, method="ox"):
    """
    Crossover dispatcher.

    Parameters
    ----------
    p1, p2 : np.ndarray
        Parent permutations
    method : str
        'ox' or 'pmx'
    """
    if method == "ox":
        return order_crossover(p1, p2)
    elif method == "pmx":
        return pmx_crossover(p1, p2)
    else:
        raise ValueError(f"Unknown crossover method: {method}")
