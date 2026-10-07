"""Plot checked-in benchmark records without rerunning experiments."""

import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    parser.add_argument("--png", action="store_true", help="Also render local PNG previews.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10})
    curves = read_rows(args.results / "knn_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fig.suptitle("KNN from scratch — paired WDBC holdouts", fontsize=16, fontweight="bold")
    normalized = [row for row in curves if row["variant"] == "normalized"]
    raw = [row for row in curves if row["variant"] == "raw"]
    for metric, label, color in (("train_resubstitution", "Training (includes query)", "#2563eb"),
                                  ("train_excluding_self", "Training (query excluded)", "#0d9488"),
                                  ("test_accuracy", "Held-out test", "#ea580c")):
        axes[0].errorbar([int(r["k"]) for r in normalized],
                         [float(r[f"{metric}_mean"]) for r in normalized],
                         yerr=[float(r[f"{metric}_std"]) for r in normalized],
                         label=label, color=color, linewidth=1.5, capsize=2)
    for group, label, color in ((normalized, "Train-fitted normalization", "#2563eb"),
                                 (raw, "Raw features", "#ea580c")):
        axes[1].errorbar([int(r["k"]) for r in group], [float(r["test_accuracy_mean"]) for r in group],
                         yerr=[float(r["test_accuracy_std"]) for r in group], label=label,
                         color=color, linewidth=1.5, capsize=2)
    for ax, title in zip(axes, ("Normalized features: training and test", "Normalization comparison: test")):
        ax.set(xlabel="Number of neighbors (k)", ylabel="Mean accuracy ± split SD", title=title, ylim=(0.82, 1.015))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(args.output / "knn.svg", metadata={"Date": None})
    if args.png:
        fig.savefig(args.output / "knn.png", dpi=150)
    plt.close(fig)
    rows = read_rows(args.results / "tree_runs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    fig.suptitle("Categorical trees from scratch — paired Car Evaluation holdouts", fontsize=16, fontweight="bold")
    for ax, variant, title in zip(axes, ("entropy", "gini", "entropy_purity85"),
                                  ("Entropy", "Gini", "Entropy, 85% purity stop")):
        group = [row for row in rows if row["variant"] == variant]
        train = [float(row["train_accuracy"]) for row in group]
        test = [float(row["test_accuracy"]) for row in group]
        ax.hist(train, bins=np.linspace(0.80, 1.00001, 21), color="#2563eb", alpha=0.65, label="Training")
        ax.hist(test, bins=np.linspace(0.80, 1.00001, 21), color="#ea580c", alpha=0.7, label="Test")
        ax.set(xlabel="Accuracy", ylabel="Number of splits", title=title, xlim=(0.80, 1.005))
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(args.output / "trees.svg", metadata={"Date": None})
    if args.png:
        fig.savefig(args.output / "trees.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
