import numpy as np


class AdaptiveController:
    def __init__(self, config):
        self.pc_min = config["pc_min"]
        self.pc_max = config["pc_max"]
        self.pm_min = config["pm_min"]
        self.pm_max = config["pm_max"]

        self.stagnation_threshold = config["stagnation_threshold"]

        self.max_fitness_std = 1e-8
        self.max_diversity = 1e-8

        self.no_improve_generations = 0
        self.best_fitness_so_far = -np.inf

    def update_history(self, metrics):
        self.max_fitness_std = max(
            self.max_fitness_std, metrics["fitness_std"]
        )
        self.max_diversity = max(
            self.max_diversity, metrics["diversity"]
        )

        if metrics["best_fitness"] > self.best_fitness_so_far:
            self.best_fitness_so_far = metrics["best_fitness"]
            self.no_improve_generations = 0
        else:
            self.no_improve_generations += 1

    def get_pc(self, metrics):
        ratio = metrics["fitness_std"] / (self.max_fitness_std + 1e-8)
        pc = self.pc_min + (self.pc_max - self.pc_min) * (1 - ratio)
        return float(np.clip(pc, self.pc_min, self.pc_max))

    def get_pm(self):
        stagnation = min(
            self.no_improve_generations / self.stagnation_threshold, 1.0
        )
        pm = self.pm_min + (self.pm_max - self.pm_min) * stagnation
        return float(pm)

    def get_selection_method(self, metrics):
        ratio = metrics["diversity"] / (self.max_diversity + 1e-8)
        p_sus = 1 - ratio
        return "sus" if np.random.rand() < p_sus else "roulette"
