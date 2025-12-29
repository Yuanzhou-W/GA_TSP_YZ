# ga/operators/metrics.py

import numpy as np


# --------------------------------------------------
# Fitness evaluation
# --------------------------------------------------

def evaluate_population(population, distance_matrix):
    """
    Compute path length and fitness for each individual.

    Returns
    -------
    fitness : np.ndarray
        Fitness values (1 / length)
    lengths : np.ndarray
        Tour lengths
    """
    pop_size = len(population)
    lengths = np.zeros(pop_size)

    for i, individual in enumerate(population):
        length = 0.0
        for j in range(len(individual)):
            a = individual[j]
            b = individual[(j + 1) % len(individual)]
            length += distance_matrix[a][b]
        lengths[i] = length

    fitness = 1.0 / (lengths + 1e-12)
    return fitness, lengths


# --------------------------------------------------
# Diversity (edge-based)
# --------------------------------------------------

def compute_population_diversity(population):
    """
    Edge-based population diversity for TSP.

    Definition:
    diversity = (# unique edges) / (# population edges)

    Returns
    -------
    float
    """
    edge_set = set()
    total_edges = 0

    for ind in population:
        n = len(ind)
        for i in range(n):
            a = ind[i]
            b = ind[(i + 1) % n]
            edge = tuple(sorted((a, b)))
            edge_set.add(edge)
            total_edges += 1

    if total_edges == 0:
        return 0.0

    return len(edge_set) / total_edges


# --------------------------------------------------
# Best individual
# --------------------------------------------------

def get_best_individual(population, distance_matrix):
    """
    Get best individual and its length.
    """
    _, lengths = evaluate_population(population, distance_matrix)
    best_idx = np.argmin(lengths)
    return population[best_idx].copy(), lengths[best_idx]
