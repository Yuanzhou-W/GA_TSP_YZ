# analysis/show_route_and_convergence.py

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.tsp_loader import load_tsp


def compute_tour_length(tour, dist):
    return sum(
        dist[tour[i], tour[(i + 1) % len(tour)]]
        for i in range(len(tour))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsp", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--out",
        default="analysis/results/figures/route_and_convergence.png"
    )
    args = parser.parse_args()

    # Load data
    tsp = load_tsp(args.tsp)
    coords = tsp.coords
    dist = tsp.distance_matrix

    with open(args.result, "r", encoding="utf-8") as f:
        log = json.load(f)

    tour = np.array(log["best_individual"])
    best_length = log["summary"]["best_length"]
    strategy = log["meta"]["strategy"]

    generations = log["generations"]
    convergence = [
        1.0 / g["metrics"]["best_fitness"] for g in generations
    ]

    # ---------------- Plot ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Route ----
    ordered = np.append(tour, tour[0])
    axes[0].plot(
        coords[ordered, 0],
        coords[ordered, 1],
        "-o",
        markersize=3,
        linewidth=1
    )
    axes[0].scatter(coords[:, 0], coords[:, 1], s=10)
    axes[0].set_title(
        f"{strategy}\nBest length = {best_length:.2f}"
    )
    axes[0].axis("equal")
    axes[0].axis("off")

    # ---- Right: Convergence ----
    axes[1].plot(convergence)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Best Tour Length")
    axes[1].set_title("Convergence Curve")
    axes[1].grid(True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
