# ga/strategies/base.py

from abc import ABC, abstractmethod


class GAStrategy(ABC):
    """
    Minimal interface for GA strategies.
    Engine controls the workflow.
    Strategy controls the operators.
    """

    name = "BaseStrategy"

    def __init__(self, pc=0.9, pm=0.1):
        self.pc = pc
        self.pm = pm

        # for logging & analysis
        self.last_selection_method = None

    # --------------------------------------------------
    # Required by GAEngine
    # --------------------------------------------------

    @abstractmethod
    def evaluate(self, population, distance_matrix):
        """
        Return:
            fitness: np.ndarray
            lengths: np.ndarray
        """
        pass

    @abstractmethod
    def evolve(self, population, fitness, generation):
        """
        Return:
            new_population: np.ndarray
        """
        pass

    @abstractmethod
    def compute_diversity(self, population):
        """
        Return:
            diversity: float
        """
        pass
