# ga/engine.py

import time
import numpy as np


class GAEngine:
    """
    Strategy-driven Genetic Algorithm Engine.

    Compatible with:
    - tsp / distance_matrix
    - population_size / pop_size
    - generations / max_generations
    - elite_size
    - run(verbose=...)
    """

    def __init__(
        self,
        tsp=None,
        distance_matrix=None,
        strategy=None,
        population_size=None,
        pop_size=None,
        generations=None,
        max_generations=None,
        elite_size=None,
        seed=None,
        verbose=True,
    ):
        # --------------------------------------------------
        # Basic checks
        # --------------------------------------------------
        if tsp is None and distance_matrix is None:
            raise ValueError(
                "GAEngine requires either tsp or distance_matrix."
            )

        if strategy is None:
            raise ValueError("GAEngine requires a strategy.")

        # --------------------------------------------------
        # TSP / distance matrix compatibility
        # --------------------------------------------------
        if tsp is not None:
            self.distance_matrix = np.asarray(tsp.distance_matrix)
            self.n_cities = tsp.num_cities
            self.tsp_name = tsp.name
        else:
            self.distance_matrix = np.asarray(distance_matrix)
            self.n_cities = self.distance_matrix.shape[0]
            self.tsp_name = "Unknown-TSP"

        # --------------------------------------------------
        # Population size compatibility
        # --------------------------------------------------
        if population_size is None and pop_size is None:
            self.population_size = 100
        elif population_size is None:
            self.population_size = pop_size
        else:
            self.population_size = population_size

        # --------------------------------------------------
        # Generation compatibility
        # --------------------------------------------------
        if generations is None and max_generations is None:
            self.generations = 500
        elif generations is None:
            self.generations = max_generations
        else:
            self.generations = generations

        # --------------------------------------------------
        # Elitism (record only)
        # --------------------------------------------------
        self.elite_size = elite_size

        self.strategy = strategy
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        self.population = self._init_population()
        self.best_individual = None
        self.best_length = np.inf

        # --------------------------------------------------
        # Unified logs
        # --------------------------------------------------
        self.logs = {
            "meta": {
                "tsp": self.tsp_name,
                "n_cities": self.n_cities,
                "strategy": strategy.name,
                "population_size": self.population_size,
                "generations": self.generations,
                "elite_size": self.elite_size,
            },
            "history": {
                "best_length": [],
                "mean_length": [],
                "fitness_std": [],
                "diversity": [],
                "pc": [],
                "pm": [],
                "selection": [],
            },
        }

    # --------------------------------------------------
    # Population initialization
    # --------------------------------------------------

    def _init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.permutation(self.n_cities)
            population.append(individual)
        return np.array(population)

    # --------------------------------------------------
    # Main GA loop
    # --------------------------------------------------

    def run(self, verbose=None):
        """
        Run GA process.

        Parameters
        ----------
        verbose : bool or None
            If provided, overrides self.verbose for this run.
        """
        if verbose is not None:
            self.verbose = verbose

        start_time = time.time()

        for gen in range(self.generations):

            # -------- Evaluation --------
            fitness, lengths = self.strategy.evaluate(
                self.population,
                self.distance_matrix,
            )

            # -------- Best solution update --------
            idx = np.argmin(lengths)
            if lengths[idx] < self.best_length:
                self.best_length = lengths[idx]
                self.best_individual = self.population[idx].copy()

            # -------- Record statistics --------
            self._record(fitness, lengths)

            # -------- Evolution --------
            self.population = self.strategy.evolve(
                population=self.population,
            )

            if self.verbose and (gen + 1) % 50 == 0:
                print(
                    f"[Gen {gen + 1:4d}] "
                    f"Best length = {self.best_length:.2f}"
                )

        # -------- Final logs --------
        self.logs["best_individual"] = self.best_individual.tolist()
        self.logs["best_length"] = self.best_length
        self.logs["runtime"] = time.time() - start_time

        return self.best_individual, self.logs

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _record(self, fitness, lengths):
        history = self.logs["history"]

        history["best_length"].append(np.min(lengths))
        history["mean_length"].append(np.mean(lengths))
        history["fitness_std"].append(np.std(fitness))

        history["diversity"].append(
            self.strategy.compute_diversity(self.population)
        )

        history["pc"].append(float(self.strategy.pc))
        history["pm"].append(float(self.strategy.pm))
        history["selection"].append(
            self.strategy.last_selection_method
        )
