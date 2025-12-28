import numpy as np


def compute_fitness(population, tsp):
    fitness = []
    for individual in population:
        length = tsp.evaluate(individual)
        fitness.append(1.0 / (length + 1e-8))
    return np.array(fitness)


def compute_diversity(population):
    n = len(population)
    if n <= 1:
        return 0.0

    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.sum(population[i] != population[j])
            count += 1

    return total / count
