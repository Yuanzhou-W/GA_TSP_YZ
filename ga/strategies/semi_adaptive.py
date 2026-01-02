# ga/strategies/semi_adaptive.py

import numpy as np

from ga.strategies.base import GAStrategy
from ga.operators.selection import select
from ga.operators.crossover import crossover
from ga.operators.mutation import mutate
from ga.operators.metrics import (
    evaluate_population,
    compute_population_diversity,
)


class SemiAdaptiveGAStrategy(GAStrategy):
    """
    Semi-Adaptive GA:
    - Selection method fixed (roulette / sus)
    - pc / pm adapt based on population diversity
    """

    name = "SemiAdaptiveGA"

    def __init__(self, config):
        super().__init__()

        # --------------------------------------------------
        # base pc / pm (用于初始值)
        # --------------------------------------------------
        pc_cfg = config.get("pc", 0.9)
        pm_cfg = config.get("pm", 0.02)

        self.pc_base = (
            pc_cfg["max"] if isinstance(pc_cfg, dict) else float(pc_cfg)
        )
        self.pm_base = (
            pm_cfg["min"] if isinstance(pm_cfg, dict) else float(pm_cfg)
        )

        # --------------------------------------------------
        # adaptive bounds（⚠️ 你刚才缺的就是这四个）
        # --------------------------------------------------
        self.pc_min = 0.6
        self.pc_max = 0.95
        self.pm_min = 0.01
        self.pm_max = 0.3

        # --------------------------------------------------
        # current parameters
        # --------------------------------------------------
        self.pc = self.pc_base
        self.pm = self.pm_base

        # --------------------------------------------------
        # operators
        # --------------------------------------------------
        self.selection_method = config.get("selection_method", "roulette")
        self.crossover_method = config.get("crossover_method", "ox")
        self.mutation_method = config.get("mutation_method", "swap")

        self.last_diversity = None
        self.last_selection_method = self.selection_method
    # --------------------------------------------------
    def evaluate(self, population, distance_matrix):
        return evaluate_population(population, distance_matrix)
    # --------------------------------------------------
    def update_parameters(self, diversity):
        """
        Semi-adaptive rule:
        - low diversity  -> increase mutation
        - high diversity -> increase crossover
        """
        self.last_diversity = diversity

        # diversity assumed in [0, 1]
        self.pc = self.pc_min + diversity * (self.pc_max - self.pc_min)
        self.pm = self.pm_max - diversity * (self.pm_max - self.pm_min)

        self.pc = float(np.clip(self.pc, self.pc_min, self.pc_max))
        self.pm = float(np.clip(self.pm, self.pm_min, self.pm_max))

    # --------------------------------------------------
    def evolve(self, population, distance_matrix, elite_size):
        pop_size = len(population)

        fitness, lengths = self.evaluate(population, distance_matrix)
        diversity = compute_population_diversity(population)

        # 🔴 真正起作用的地方
        self.update_parameters(diversity)

        parents = select(
            fitness,
            num_selected=pop_size,
            method=self.selection_method,
        )
        self.last_selection_method = self.selection_method

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
        return {
            "pc": self.pc,
            "pm": self.pm,
            "diversity": self.last_diversity,
        }
    # --------------------------------------------------
    def compute_diversity(self, population):
        """
        Compute population diversity.
        This method is required by GAStrategy abstract interface.
        """
        return compute_population_diversity(population)
