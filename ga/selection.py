import numpy as np


def normalize_fitness(fitness, eps=1e-12):
    fitness = np.asarray(fitness, dtype=np.float64)
    fitness = np.maximum(fitness, eps)
    total = np.sum(fitness)
    if total == 0:
        return np.ones_like(fitness) / len(fitness)
    return fitness / total


def roulette_wheel_selection(fitness, num_selected):
    prob = normalize_fitness(fitness)
    cumulative = np.cumsum(prob)
    r = np.random.rand(num_selected)
    indices = np.searchsorted(cumulative, r)

    return indices, {
        "method": "roulette",
        "probabilities": prob.tolist(),
        "cumulative": cumulative.tolist()
    }


def stochastic_universal_sampling(fitness, num_selected):
    prob = normalize_fitness(fitness)
    cumulative = np.cumsum(prob)

    step = 1.0 / num_selected
    start = np.random.uniform(0, step)
    pointers = start + step * np.arange(num_selected)

    indices = np.searchsorted(cumulative, pointers)

    return indices, {
        "method": "sus",
        "probabilities": prob.tolist(),
        "pointers": pointers.tolist()
    }


def select(fitness, num_selected, method, generation=None):
    if method == "roulette":
        indices, stats = roulette_wheel_selection(fitness, num_selected)
    elif method == "sus":
        indices, stats = stochastic_universal_sampling(fitness, num_selected)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    stats.update({
        "generation": generation,
        "num_selected": num_selected,
        "fitness_mean": float(np.mean(fitness)),
        "fitness_std": float(np.std(fitness))
    })

    return indices, stats
