# ga/strategies/adaptive.py

import numpy as np

from ga.strategies.base import GAStrategy
from ga.operators.selection import select
from ga.operators.crossover import crossover
from ga.operators.mutation import mutate
from ga.operators.metrics import (
    evaluate_population,
    compute_population_diversity,
)


class AdaptiveGAStrategy(GAStrategy):
    """
    Fully Adaptive GA:
    - Adaptive selection (roulette + SUS mix)
    - Adaptive Pc / Pm
    """

    name = "AdaptiveGA"

    def __init__(self, config):
        super().__init__(config)

        # --------------------------------------------------
        # pc / pm config compatibility
        # --------------------------------------------------
        pc_cfg = config.get("pc", 0.9)
        pm_cfg = config.get("pm", 0.02)

        if isinstance(pc_cfg, dict):
            self.pc_min = pc_cfg["min"]
            self.pc_max = pc_cfg["max"]
        else:
            self.pc_min = self.pc_max = pc_cfg

        if isinstance(pm_cfg, dict):
            self.pm_min = pm_cfg["min"]
            self.pm_max = pm_cfg["max"]
        else:
            self.pm_min = self.pm_max = pm_cfg

        # --------------------------------------------------
        # adaptive selection parameters
        # --------------------------------------------------
        self.sus_ratio = 0.5
        self.sus_ratio_min = 0.1
        self.sus_ratio_max = 0.9

        self.crossover_method = config.get("crossover_method", "ox")
        self.mutation_method = config.get("mutation_method", "swap")

        # init
        self.pc = self.pc_max
        self.pm = self.pm_min
        self.last_diversity = None

    # --------------------------------------------------
    def evaluate(self, population, distance_matrix):
        return evaluate_population(population, distance_matrix)

    # --------------------------------------------------
    def compute_diversity(self, population):
        return compute_population_diversity(population)

    # --------------------------------------------------
    def update_parameters(self, diversity, gen, max_gen):
        """
        Adaptive Pc / Pm & selection ratio
        """
        self.last_diversity = diversity

        # ---- Pc / Pm driven by diversity ----
        self.pc = self.pc_min + (self.pc_max - self.pc_min) * diversity
        self.pm = self.pm_max - (self.pm_max - self.pm_min) * diversity

        # ---- generation annealing ----
        t = gen / max_gen
        self.pc *= (1.0 - 0.4 * t)
        self.pm *= (1.0 + 0.4 * t)

        self.pc = np.clip(self.pc, self.pc_min, self.pc_max)
        self.pm = np.clip(self.pm, self.pm_min, self.pm_max)

        # ---- adaptive selection mixing ----
        # lower diversity -> more SUS
        self.sus_ratio = 1.0 - diversity
        self.sus_ratio = np.clip(
            self.sus_ratio,
            self.sus_ratio_min,
            self.sus_ratio_max,
        )

    # --------------------------------------------------
    def mixed_selection(self, fitness, pop_size):
        """
        Mix roulette and SUS selection
        """
        n_sus = int(pop_size * self.sus_ratio)
        n_roulette = pop_size - n_sus

        parents = []

        if n_roulette > 0:
            parents.extend(
                select(
                    fitness,
                    num_selected=n_roulette,
                    method="roulette",
                )
            )

        if n_sus > 0:
            parents.extend(
                select(
                    fitness,
                    num_selected=n_sus,
                    method="sus",
                )
            )

        return np.array(parents)

    # --------------------------------------------------
    def evolve(self, population, distance_matrix, elite_size):
        pop_size = len(population)

        fitness, lengths = self.evaluate(population, distance_matrix)
        diversity = self.compute_diversity(population)

        self.update_parameters(
            diversity,
            self.generation,
            self.max_generations,
        )

        parents = self.mixed_selection(fitness, pop_size)

        # ---- Elitism ----
        elite_idx = np.argsort(lengths)[:elite_size]
        new_population = [population[i].copy() for i in elite_idx]

        # ---- Offspring ----
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
        Record adaptive parameters
        """
        return {
            "pc": self.pc,
            "pm": self.pm,
            "diversity": self.last_diversity,
            "sus_ratio": self.sus_ratio,
        }
