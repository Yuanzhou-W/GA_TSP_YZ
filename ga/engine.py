import numpy as np
from ga.selection import select
from ga.crossover import crossover
from ga.mutation import mutate
from ga.adaptive import AdaptiveController
from ga.metrics import compute_fitness, compute_diversity


class GeneticAlgorithm:
    def __init__(self, config, tsp):
        self.config = config
        self.tsp = tsp

        self.pop_size = config["population_size"]
        self.max_generations = config["max_generations"]

        self.selection_method = None
        self.crossover_method = config["crossover_method"]
        self.mutation_method = config["mutation_method"]

        self.adaptive = AdaptiveController(config["adaptive"])

        self.logs = {
            "fitness": [],
            "diversity": [],
            "parameters": [],
            "selection": [],
            "crossover": [],
            "mutation": []
        }

        self.population = self._init_population()

    def _init_population(self):
        n = self.tsp.num_cities
        return np.array([np.random.permutation(n) for _ in range(self.pop_size)])

    def step(self, generation):
        fitness = compute_fitness(self.population, self.tsp)
        diversity = compute_diversity(self.population)

        metrics = {
            "best_fitness": float(np.max(fitness)),
            "mean_fitness": float(np.mean(fitness)),
            "fitness_std": float(np.std(fitness)),
            "diversity": float(diversity)
        }

        self.adaptive.update_history(metrics)

        pc = self.adaptive.get_pc(metrics)
        pm = self.adaptive.get_pm()
        sel_method = self.adaptive.get_selection_method(metrics)

        self.logs["parameters"].append({
            "generation": generation,
            "pc": pc,
            "pm": pm,
            "selection_method": sel_method
        })

        idx, sel_stats = select(
            fitness, self.pop_size, sel_method, generation
        )
        self.logs["selection"].append(sel_stats)

        mating_pool = self.population[idx]
        new_population = []

        for i in range(0, self.pop_size, 2):
            p1 = mating_pool[i]
            p2 = mating_pool[(i + 1) % self.pop_size]

            c1, c2, cross_stats = crossover(
                p1, p2, pc, self.crossover_method, generation
            )
            self.logs["crossover"].append(cross_stats)

            c1, m1 = mutate(c1, pm, self.mutation_method, generation)
            c2, m2 = mutate(c2, pm, self.mutation_method, generation)

            self.logs["mutation"].extend([m1, m2])
            new_population.extend([c1, c2])

        self.population = np.array(new_population[:self.pop_size])

        self.logs["fitness"].append(metrics)
        self.logs["diversity"].append({
            "generation": generation,
            "diversity": diversity
        })

    def run(self):
        for gen in range(self.max_generations):
            self.step(gen)
        return self.population, self.logs
