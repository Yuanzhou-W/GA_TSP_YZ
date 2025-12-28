from ga.engine import GeneticAlgorithm
from utils.tsp_loader import load_tsp
import json


def main():
    tsp = load_tsp("./data/ch130.tsp")

    config = {
        "population_size": 100,
        "max_generations": 300,
        "selection_methods": ["roulette", "sus"],
        "crossover_method": "ox",
        "mutation_method": "inversion",
        "adaptive": {
            "pc_min": 0.6,
            "pc_max": 0.95,
            "pm_min": 0.01,
            "pm_max": 0.3,
            "stagnation_threshold": 30
        }
    }

    ga = GeneticAlgorithm(config, tsp)
    _, logs = ga.run()

    with open("results/run_log.json", "w") as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    main()
