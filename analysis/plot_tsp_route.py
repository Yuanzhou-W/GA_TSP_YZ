# analysis/plot_tsp_route.py

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


# --------------------------------------------------
# Utility
# --------------------------------------------------

def compute_tour_length(tour, distance_matrix):
    length = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        length += distance_matrix[a, b]
    return length


# --------------------------------------------------
# Plot
# --------------------------------------------------

def plot_tsp_route(
    coords,
    tour,
    distance_matrix,
    title,
    save_path=None
):
    x = coords[:, 0]
    y = coords[:, 1]

    ordered = np.append(tour, tour[0])
    route_x = x[ordered]
    route_y = y[ordered]

    length = compute_tour_length(tour, distance_matrix)

    plt.figure(figsize=(8, 8))
    plt.plot(route_x, route_y, "-o", markersize=3, linewidth=1)
    plt.scatter(x, y, s=10)

    plt.title(f"{title}\nBest tour length = {length:.2f}")
    plt.axis("equal")
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot best TSP route from GA result"
    )
    parser.add_argument("--tsp", required=True, help="Path to .tsp file")
    parser.add_argument("--result", required=True, help="GA run json file")
    parser.add_argument(
        "--out",
        default="analysis/results/figures/best_route.png",
        help="Output image path"
    )

    args = parser.parse_args()

    # Load TSP
    tsp = load_tsp(args.tsp)
    coords = np.asarray(tsp.coords)          # 🔧 fix
    distance_matrix = np.asarray(tsp.distance_matrix)

    # Load GA result
    with open(args.result, "r", encoding="utf-8") as f:
        log = json.load(f)

    best_individual = log.get("best_individual")
    if best_individual is None:
        raise ValueError("best_individual not found in result file.")

    tour = np.array(best_individual, dtype=int)

    title = (
        f"{tsp.name} - {log['meta']['strategy']}\n"
        f"Run {log['meta']['run_id']}"
    )

    plot_tsp_route(
        coords=coords,
        tour=tour,
        distance_matrix=distance_matrix,
        title=title,
        save_path=args.out
    )


if __name__ == "__main__":
    main()
