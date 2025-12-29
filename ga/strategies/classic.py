# ga/strategies/classic.py

import numpy as np

from ga.strategies.base import GAStrategy
from ga.operators.selection import select
from ga.operators.crossover import crossover
from ga.operators.mutation import mutate
from ga.operators.metrics import (
    evaluate_population,
    compute_population_diversity,
)


class ClassicGAStrategy(GAStrategy):
    """
    Classic GA: Roulette selection + fixed Pc/Pm
    """

    name = "ClassicGA"

    def __init__(self, config):
        super().__init__(config)

        # ---- 固定参数（强制 scalar）----
        self.pc = (
            config["pc"]["max"]
            if isinstance(config["pc"], dict)
            else float(config["pc"])
        )
        self.pm = (
            config["pm"]["max"]
            if isinstance(config["pm"], dict)
            else float(config["pm"])
        )

        self.selection_method = "roulette"
        self.crossover_method = config.get("crossover_method", "ox")
        self.mutation_method = config.get("mutation_method", "swap")

        self.last_selection_method = self.selection_method

    # --------------------------------------------------
    def evaluate(self, population, distance_matrix):
        return evaluate_population(population, distance_matrix)

    # --------------------------------------------------
    def compute_diversity(self, population):
        return compute_population_diversity(population)

    # --------------------------------------------------
    def evolve(self, population, distance_matrix, elite_size):
        pop_size = len(population)

        fitness, lengths = self.evaluate(population, distance_matrix)

        # ---- Selection (indices!) ----
        parent_indices = select(
            fitness,
            method=self.selection_method,
            num_selected=pop_size,
        )
        self.last_selection_method = self.selection_method

        # ---- Elitism ----
        elite_size = elite_size or 0
        elite_idx = np.argsort(lengths)[:elite_size]
        new_population = [population[i].copy() for i in elite_idx]

        # ---- Generate offspring ----
        i = 0
        while len(new_population) < pop_size:
            p1 = population[parent_indices[i % pop_size]]
            p2 = population[parent_indices[(i + 1) % pop_size]]
            i += 2

            if np.random.rand() < self.pc:
                c1, c2 = crossover(
                    p1,
                    p2,
                    method=self.crossover_method,
                )
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate(c1, self.pm, method=self.mutation_method)
            c2 = mutate(c2, self.pm, method=self.mutation_method)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        return np.array(new_population)

    # --------------------------------------------------
    def record(self):
        return {
            "pc": self.pc,
            "pm": self.pm,
            "selection": self.selection_method,
        }
