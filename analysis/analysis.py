# analysis/analysis.py

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Path configuration (robust, OS-independent)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RESULT_ROOT = os.path.join(PROJECT_ROOT, "experiment_results", "experiments")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "experiment_results", "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)



# --------------------------------------------------
# Data Loading
# --------------------------------------------------

def load_all_results():
    """
    Load all GA experiment experiment_results.
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
        curves = []

        for log in runs:
            # ★ 关键：用 history.best_length
            history = log.get("history", {})
            best_len = history.get("best_length", [])

            if len(best_len) == 0:
                continue

            curves.append(best_len)

        if not curves:
            print(f"[Skip] {strategy} has no valid history")
            continue

        # 对齐长度
        min_len = min(len(c) for c in curves)
        curves = np.array([c[:min_len] for c in curves])

        mean_curve = curves.mean(axis=0)

        plt.plot(mean_curve, label=strategy)

    plt.xlabel("Generation")
    plt.ylabel("Best Tour Length")
    plt.title("GA Convergence Comparison (ch130)")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(FIGURE_DIR, "fitness_convergence.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


# --------------------------------------------------
# Plot 2: Stability (boxplot)
# --------------------------------------------------

def plot_stability(data):
    labels = []
    values = []

    for strategy, runs in data.items():
        final_lengths = [log["best_length"] for log in runs]
        labels.append(strategy)
        values.append(final_lengths)

    plt.figure(figsize=(8, 6))
    plt.boxplot(values, tick_labels=labels, showmeans=True)
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
        times = [log["runtime"] for log in runs]
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

def plot_adaptive_dynamics(data):
    """
    Plot Pc / Pm / Diversity evolution for adaptive GA strategies.
    """
    plt.figure(figsize=(10, 6))

    for strategy, runs in data.items():

        # 只对自适应类策略画图
        if "Adaptive" not in strategy:
            continue

        # --- collect curves ---
        pc_curves = []
        pm_curves = []
        div_curves = []

        for log in runs:
            history = log["history"]
            pc_curves.append(history["pc"])
            pm_curves.append(history["pm"])
            div_curves.append(history["diversity"])

        # --- align length ---
        min_len = min(
            min(len(pc) for pc in pc_curves),
            min(len(pm) for pm in pm_curves),
            min(len(dv) for dv in div_curves),
        )

        pc = np.mean([c[:min_len] for c in pc_curves], axis=0)
        pm = np.mean([c[:min_len] for c in pm_curves], axis=0)
        dv = np.mean([c[:min_len] for c in div_curves], axis=0)

        x = np.arange(min_len)

        # --- plotting ---
        plt.plot(x, pc, label=f"{strategy} Pc", linestyle="-")
        plt.plot(x, pm, label=f"{strategy} Pm", linestyle="--")
        plt.plot(x, dv, label=f"{strategy} Diversity", linestyle=":")

    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Adaptive GA Parameter & Diversity Evolution")
    plt.legend()
    plt.grid(True)

    path = os.path.join(FIGURE_DIR, "adaptive_dynamics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    data = load_all_results()

    if not data:
        raise RuntimeError("No experiment experiment_results found.")

    plot_convergence(data)
    plot_stability(data)
    plot_runtime(data)
    plot_adaptive_dynamics(data)   # ← 新增





if __name__ == "__main__":
    main()
