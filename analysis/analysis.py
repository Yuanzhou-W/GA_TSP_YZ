# analysis/analysis.py

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILE = "../results/run_log.json"
FIG_DIR = "../results/figures"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_logs(path):
    with open(path, "r") as f:
        logs = json.load(f)
    return logs


def extract_fitness_df(logs):
    df = pd.DataFrame(logs["fitness"])
    df["generation"] = df.index
    return df


def extract_diversity_df(logs):
    df = pd.DataFrame(logs["diversity"])
    return df


def extract_parameter_df(logs):
    df = pd.DataFrame(logs["parameters"])
    return df


def extract_selection_df(logs):
    df = pd.DataFrame(logs["selection"])
    return df


# -------------------------------------------------
# Plot functions
# -------------------------------------------------

def plot_fitness_curve(df):
    plt.figure()
    plt.plot(df["generation"], df["best_fitness"], label="Best Fitness")
    plt.plot(df["generation"], df["mean_fitness"], label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "fitness_convergence.png"), dpi=300)
    plt.close()


def plot_fitness_std(df):
    plt.figure()
    plt.plot(df["generation"], df["fitness_std"])
    plt.xlabel("Generation")
    plt.ylabel("Fitness Std")
    plt.title("Fitness Standard Deviation")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "fitness_std.png"), dpi=300)
    plt.close()


def plot_diversity(df):
    plt.figure()
    plt.plot(df["generation"], df["diversity"])
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Population Diversity")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "diversity.png"), dpi=300)
    plt.close()


def plot_parameters(df):
    plt.figure()
    plt.plot(df["generation"], df["pc"], label="Pc")
    plt.plot(df["generation"], df["pm"], label="Pm")
    plt.xlabel("Generation")
    plt.ylabel("Probability")
    plt.title("Adaptive Parameters")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "adaptive_parameters.png"), dpi=300)
    plt.close()


def plot_selection_usage(df):
    counts = df["method"].value_counts()

    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xlabel("Selection Method")
    plt.ylabel("Usage Count")
    plt.title("Selection Method Usage")
    plt.grid(axis="y")
    plt.savefig(os.path.join(FIG_DIR, "selection_usage.png"), dpi=300)
    plt.close()


def plot_selection_over_time(df):
    df_time = df[["generation", "method"]].copy()
    df_time["sus"] = (df_time["method"] == "sus").astype(int)

    sus_ratio = df_time.groupby("generation")["sus"].mean()

    plt.figure()
    plt.plot(sus_ratio.index, sus_ratio.values)
    plt.xlabel("Generation")
    plt.ylabel("SUS Ratio")
    plt.title("SUS Usage Over Time")
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "sus_ratio_over_time.png"), dpi=300)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ensure_dir(FIG_DIR)

    logs = load_logs(RESULT_FILE)

    fitness_df = extract_fitness_df(logs)
    diversity_df = extract_diversity_df(logs)
    param_df = extract_parameter_df(logs)
    selection_df = extract_selection_df(logs)

    plot_fitness_curve(fitness_df)
    plot_fitness_std(fitness_df)
    plot_diversity(diversity_df)
    plot_parameters(param_df)
    plot_selection_usage(selection_df)
    plot_selection_over_time(selection_df)

    print("[INFO] Analysis complete. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
