# analysis/analysis.py

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = "results/experiments"
FIGURE_DIR = "analysis/results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


# --------------------------------------------------
# Data Loading
# --------------------------------------------------

def load_all_results():
    """
    Load all GA experiment results.
    Returns:
        data[strategy] = list of run_logs
    """
    data = defaultdict(list)

    for strategy_name in os.listdir(RESULT_ROOT):
        strategy_dir = os.path.join(RESULT_ROOT, strategy_name)
        if not os.path.isdir(strategy_dir):
            continue

        for fname in os.listdir(strategy_dir):
            if not fname.endswith(".json"):
                continue

            with open(os.path.join(strategy_dir, fname), "r", encoding="utf-8") as f:
                log = json.load(f)
                data[strategy_name].append(log)

    return data


# --------------------------------------------------
# Plot 1: Convergence curves
# --------------------------------------------------

def plot_convergence(data):
    plt.figure(figsize=(8, 6))

    for strategy, runs in data.items():
        # Average best length over runs
        all_curves = []

        for log in runs:
            best_lengths = [
                1.0 / g["metrics"]["best_fitness"]
                for g in log["generations"]
            ]
            all_curves.append(best_lengths)

        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean_curve = curves.mean(axis=0)

        plt.plot(mean_curve, label=strategy)

    plt.xlabel("Generation")
    plt.ylabel("Best Tour Length")
    plt.title("GA Convergence Comparison (cn130)")
    plt.legend()
    plt.grid(True)

    path = os.path.join(FIGURE_DIR, "fitness_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {path}")


# --------------------------------------------------
# Plot 2: Stability (boxplot)
# --------------------------------------------------

def plot_stability(data):
    labels = []
    values = []

    for strategy, runs in data.items():
        final_lengths = [
            log["summary"]["best_length"] for log in runs
        ]
        labels.append(strategy)
        values.append(final_lengths)

    plt.figure(figsize=(8, 6))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel("Best Tour Length")
    plt.title("GA Stability Comparison (Final Solution)")
    plt.grid(axis="y")

    path = os.path.join(FIGURE_DIR, "stability_boxplot.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {path}")


# --------------------------------------------------
# Plot 3: Runtime comparison
# --------------------------------------------------

def plot_runtime(data):
    strategies = []
    runtimes = []

    for strategy, runs in data.items():
        times = [log["summary"]["runtime_sec"] for log in runs]
        strategies.append(strategy)
        runtimes.append(np.mean(times))

    plt.figure(figsize=(8, 6))
    plt.bar(strategies, runtimes)
    plt.ylabel("Runtime (seconds)")
    plt.title("GA Runtime Comparison")
    plt.grid(axis="y")

    path = os.path.join(FIGURE_DIR, "runtime_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    data = load_all_results()

    if not data:
        raise RuntimeError("No experiment results found.")

    plot_convergence(data)
    plot_stability(data)
    plot_runtime(data)


if __name__ == "__main__":
    main()
