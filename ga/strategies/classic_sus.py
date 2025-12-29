# ga/strategies/classic_sus.py

import numpy as np

from ga.strategies.base import GAStrategy
from ga.operators.selection import select
from ga.operators.crossover import crossover
from ga.operators.mutation import mutate
from ga.operators.metrics import (
    evaluate_population,
    compute_population_diversity,
)


class ClassicSUSGAStrategy(GAStrategy):
    """
    Classic GA with SUS (Stochastic Universal Sampling)
    """

    name = "ClassicGA_SUS"

    def __init__(self, config):
        super().__init__(config)

        # ---- fixed parameters (force scalar) ----
        self.pc = (
            config["pc"]["max"]
            if isinstance(config["pc"], dict)
            else config["pc"]
        )
        self.pm = (
            config["pm"]["max"]
            if isinstance(config["pm"], dict)
            else config["pm"]
        )

        self.selection_method = "sus"
        self.crossover_method = config.get("crossover_method", "ox")
        self.mutation_method = config.get("mutation_method", "swap")

    # --------------------------------------------------
    def evaluate(self, population, distance_matrix):
        """
        Fitness & path length evaluation
        """
        return evaluate_population(population, distance_matrix)

    # --------------------------------------------------
    def compute_diversity(self, population):
        """
        Population diversity (edge-based)
        """
        return compute_population_diversity(population)

    # --------------------------------------------------
    def evolve(self, population, distance_matrix, elite_size):
        """
        One generation evolution
        """
        pop_size = len(population)

        fitness, lengths = self.evaluate(population, distance_matrix)

        # ---- Selection (SUS) ----
        parents = select(
            fitness,
            num_selected=pop_size,
            method=self.selection_method,
        )

        # ---- Elitism ----
        elite_idx = np.argsort(lengths)[:elite_size]
        new_population = [population[i].copy() for i in elite_idx]

        # ---- Generate offspring ----
        i = 0
        while len(new_population) < pop_size:
            p1 = population[parents[i % pop_size]]
            p2 = population[parents[(i + 1) % pop_size]]
            i += 2

            if np.random.rand() < self.pc:
                c1, c2 = crossover(
                    p1,
                    p2,
                    pc=1.0,
                    method=self.crossover_method,
                )
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate(c1, self.pm, method=self.mutation_method)
            c2 = mutate(c2, self.pm, method=self.mutation_method)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        return new_population

    # --------------------------------------------------
    def record(self):
        """
        Record fixed parameters
        """
        return {
            "pc": self.pc,
            "pm": self.pm,
            "selection": self.selection_method,
        }
