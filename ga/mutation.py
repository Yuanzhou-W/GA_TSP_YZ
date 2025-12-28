import numpy as np


def swap_mutation(individual):
    n = len(individual)
    i, j = np.random.choice(n, 2, replace=False)
    mutant = individual.copy()
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant, {"type": "swap", "positions": (int(i), int(j))}


def inversion_mutation(individual):
    n = len(individual)
    i, j = np.sort(np.random.choice(n, 2, replace=False))
    mutant = individual.copy()
    mutant[i:j + 1] = mutant[i:j + 1][::-1]
    return mutant, {
        "type": "inversion",
        "segment": (int(i), int(j)),
        "length": int(j - i + 1)
    }


def mutate(individual, pm, method, generation=None):
    stats = {
        "generation": generation,
        "pm": pm,
        "method": method,
        "mutated": False
    }

    if np.random.rand() > pm:
        return individual.copy(), stats

    if method == "swap":
        mutant, detail = swap_mutation(individual)
    elif method == "inversion":
        mutant, detail = inversion_mutation(individual)
    else:
        raise ValueError(f"Unknown mutation method: {method}")

    stats["mutated"] = True
    stats.update(detail)

    return mutant, stats
