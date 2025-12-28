import numpy as np


def ox_crossover(parent1, parent2):
    n = len(parent1)
    i, j = np.sort(np.random.choice(n, 2, replace=False))

    child = [-1] * n
    child[i:j + 1] = parent1[i:j + 1]

    pos = (j + 1) % n
    for city in parent2:
        if city not in child:
            child[pos] = city
            pos = (pos + 1) % n

    return np.array(child), {
        "type": "ox",
        "segment": (int(i), int(j)),
        "segment_length": int(j - i + 1)
    }


def pmx_crossover(parent1, parent2):
    n = len(parent1)
    i, j = np.sort(np.random.choice(n, 2, replace=False))

    child = parent1.copy()
    mapping = {}

    for k in range(i, j + 1):
        child[k] = parent2[k]
        mapping[parent2[k]] = parent1[k]

    for k in range(n):
        if i <= k <= j:
            continue
        while child[k] in mapping:
            child[k] = mapping[child[k]]

    return child, {
        "type": "pmx",
        "segment": (int(i), int(j)),
        "mapping_size": len(mapping)
    }


def crossover(parent1, parent2, pc, method, generation=None):
    if np.random.rand() > pc:
        return parent1.copy(), parent2.copy(), {
            "generation": generation,
            "pc": pc,
            "method": method,
            "crossover": False
        }

    if method == "ox":
        c1, d1 = ox_crossover(parent1, parent2)
        c2, d2 = ox_crossover(parent2, parent1)
    elif method == "pmx":
        c1, d1 = pmx_crossover(parent1, parent2)
        c2, d2 = pmx_crossover(parent2, parent1)
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return c1, c2, {
        "generation": generation,
        "pc": pc,
        "method": method,
        "crossover": True,
        "child1": d1,
        "child2": d2
    }
