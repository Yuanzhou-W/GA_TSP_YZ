# analysis/compare_routes_multi_ga.py

import argparse
import json
import os
import sys

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


COLORS = [
    "#d62728",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare best TSP routes from multiple GA strategies"
    )
    parser.add_argument("--tsp", required=True)
    parser.add_argument(
        "--experiment_results",
        nargs="+",
        required=True,
        help="List of GA result json files"
    )
    parser.add_argument(
        "--out",
        default="analysis/experiment_results/figures/route_comparison.png"
    )
    args = parser.parse_args()

    tsp = load_tsp(args.tsp)
    coords = np.asarray(tsp.coords)   # ✅ 必改


    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=12, c="black")

    for idx, path in enumerate(args.experiment_results):
        with open(path, "r", encoding="utf-8") as f:
            log = json.load(f)

        tour = np.array(log["best_individual"])
        length = log["best_length"]
        strategy = log["meta"]["strategy"]

        ordered = np.append(tour, tour[0])
        plt.plot(
            coords[ordered, 0],
            coords[ordered, 1],
            color=COLORS[idx % len(COLORS)],
            alpha=0.55,
            linewidth=1.5,
            label=f"{strategy} ({length:.1f})"
        )

    plt.title("Best Routes Comparison on cn130")
    plt.axis("equal")
    plt.axis("off")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
