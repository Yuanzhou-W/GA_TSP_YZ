# analysis/compare_edge_frequency_multi_ga.py

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Fix import path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from utils.tsp_loader import load_tsp


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def extract_edges(tour):
    """Return undirected edges from a TSP tour"""
    edges = []
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        edges.append(tuple(sorted((a, b))))
    return edges


def load_edge_frequency(result_dir, max_runs=None):
    edge_count = defaultdict(int)

    files = sorted(
        f for f in os.listdir(result_dir)
        if f.endswith(".json")
    )

    if max_runs:
        files = files[:max_runs]

    for fname in files:
        with open(os.path.join(result_dir, fname), "r", encoding="utf-8") as f:
            log = json.load(f)

        tour = np.asarray(log["best_individual"], dtype=int)
        for edge in extract_edges(tour):
            edge_count[edge] += 1

    strategy_name = os.path.basename(result_dir)
    return edge_count, len(files), strategy_name


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_edge_frequency_subplot(
    ax,
    coords,
    edge_count,
    n_runs,
    title
):
    max_count = max(edge_count.values())

    # Plot cities
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="black")

    # Plot edges
    for (a, b), count in edge_count.items():
        x = [coords[a, 0], coords[b, 0]]
        y = [coords[a, 1], coords[b, 1]]

        ratio = count / max_count
        ax.plot(
            x,
            y,
            color="tab:red",
            alpha=ratio,
            linewidth=0.5 + 3.0 * ratio
        )

    ax.set_title(f"{title}\n({n_runs} runs)")
    ax.axis("equal")
    ax.axis("off")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare edge frequency stability across GA strategies"
    )
    parser.add_argument("--tsp", required=True)
    parser.add_argument(
        "--experiment_results",
        nargs="+",
        required=True,
        help="List of strategy result directories"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display names"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Limit number of runs per algorithm"
    )
    parser.add_argument(
        "--out",
        default="analysis/experiment_results/figures/edge_frequency_comparison.png"
    )
    args = parser.parse_args()

    tsp = load_tsp(args.tsp)
    coords = np.asarray(tsp.coords)

    n_algo = len(args.experiment_results)
    ncols = 2
    nrows = (n_algo + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 6 * nrows)
    )
    axes = axes.flatten()

    for i, result_dir in enumerate(args.experiment_results):
        edge_count, n_runs, strategy_name = load_edge_frequency(
            result_dir, args.n_runs
        )

        title = (
            args.labels[i]
            if args.labels else strategy_name
        )

        plot_edge_frequency_subplot(
            axes[i],
            coords,
            edge_count,
            n_runs,
            title
        )

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Edge Frequency Stability Comparison (cn130)",
        fontsize=14
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
