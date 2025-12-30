# analysis/path_edge_frequency.py

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


def extract_edges(tour):
    edges = []
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        edges.append(tuple(sorted((a, b))))
    return edges


def main():
    parser = argparse.ArgumentParser(
        description="Edge frequency stability visualization"
    )
    parser.add_argument("--tsp", required=True)
    parser.add_argument("--experiment_results", required=True)
    parser.add_argument("--n_runs", type=int, default=None)
    parser.add_argument(
        "--out",
        default="analysis/experiment_results/figures/edge_frequency.png"
    )
    args = parser.parse_args()

    tsp = load_tsp(args.tsp)
    coords = np.asarray(tsp.coords)   # 🔧 fix

    files = sorted(
        f for f in os.listdir(args.experiment_results)
        if f.endswith(".json")
    )
    if args.n_runs:
        files = files[:args.n_runs]

    edge_count = defaultdict(int)
    strategy_name = None

    for fname in files:
        with open(os.path.join(args.experiment_results, fname), "r", encoding="utf-8") as f:
            log = json.load(f)

        if strategy_name is None:
            strategy_name = log["meta"]["strategy"]

        tour = np.array(log["best_individual"])
        edges = extract_edges(tour)

        for e in edges:
            edge_count[e] += 1

    max_count = max(edge_count.values())

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=12, c="black")

    for (a, b), count in edge_count.items():
        x = [coords[a, 0], coords[b, 0]]
        y = [coords[a, 1], coords[b, 1]]

        plt.plot(
            x,
            y,
            color="tab:red",
            alpha=count / max_count,
            linewidth=0.5 + 3.0 * (count / max_count)
        )

    plt.title(
        f"Edge Frequency Stability\n{strategy_name} ({len(files)} runs)"
    )
    plt.axis("equal")
    plt.axis("off")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
