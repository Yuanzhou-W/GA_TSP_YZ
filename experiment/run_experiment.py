# experiment/run_experiment.py

import json
import os
from datetime import datetime

from ga.engine import GAEngine
from ga.strategies.classic import ClassicGAStrategy
from ga.strategies.classic_sus import ClassicSUSGAStrategy
from ga.strategies.semi_adaptive import SemiAdaptiveGAStrategy
from ga.strategies.adaptive import AdaptiveGAStrategy
from utils.tsp_loader import load_tsp

# --------------------------------------------------
# Global experiment configuration
# --------------------------------------------------

TSP_PATH = "../data/ch130.tsp"
RESULT_ROOT = "../experiment_results/experiments"

POP_SIZE = 100
MAX_GENERATIONS = 500
ELITE_SIZE = 1
SEED = 42

N_RUNS = 10  # 可以改成 10 / 30

# --------------------------------------------------
# Strategy configurations
# --------------------------------------------------

classic_config = {
    "pc": 0.9,
    "pm": 0.02
}

adaptive_config = {
    "pc": {"min": 0.4, "max": 0.95},
    "pm": {"min": 0.01, "max": 0.4},
    "stagnation_threshold": 30,
    "max_generations": MAX_GENERATIONS
}

# --------------------------------------------------
# Utility
# --------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_run_log(strategy_name, run_id, log_data):
    strategy_dir = os.path.join(RESULT_ROOT, strategy_name)
    ensure_dir(strategy_dir)

    filename = f"run_{run_id:03d}.json"
    filepath = os.path.join(strategy_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"[Saved] {filepath}")


# --------------------------------------------------
# Main experiment loop
# --------------------------------------------------

def main():
    tsp = load_tsp(TSP_PATH)
    distance_matrix = tsp.distance_matrix

    # 每次 run 都新建 strategy
    strategy_factories = [
        ("ClassicGA", lambda: ClassicGAStrategy(classic_config.copy())),
        ("ClassicGA_SUS", lambda: ClassicSUSGAStrategy(classic_config.copy())),
        ("SemiAdaptiveGA", lambda: SemiAdaptiveGAStrategy({
            **adaptive_config,
            "selection_method": "roulette"
        })),
        ("AdaptiveGA", lambda: AdaptiveGAStrategy(adaptive_config.copy())),
    ]

    ensure_dir(RESULT_ROOT)

    for strategy_name, strategy_factory in strategy_factories:
        print("=" * 60)
        print(f"Running strategy: {strategy_name}")
        print("=" * 60)

        for run_id in range(1, N_RUNS + 1):
            print(f"[Run {run_id}/{N_RUNS}]")

            # 🔑 每次 run 都创建全新实例
            strategy = strategy_factory()

            engine = GAEngine(
                distance_matrix=distance_matrix,
                strategy=strategy,
                pop_size=POP_SIZE,
                max_generations=MAX_GENERATIONS,
                elite_size=ELITE_SIZE,
                seed=SEED + run_id
            )

            best_individual, logs = engine.run(verbose=True)

            # Attach metadata
            logs["meta"].update({
                "strategy": strategy_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "tsp": tsp.name,
                "num_cities": tsp.num_cities
            })

            save_run_log(strategy_name, run_id, logs)


if __name__ == "__main__":
    main()
