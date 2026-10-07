"""Plot recorded training loss and test confusion counts without retraining."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/digits-demo"))
    parser.add_argument("--output", type=Path, default=Path("results/digits-demo.svg"))
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text())
    loss = np.loadtxt(args.results / "loss.csv", delimiter=",", skiprows=1, ndmin=2)
    confusion = np.loadtxt(args.results / "confusion.csv", delimiter=",", skiprows=1, dtype=int)[:, 1:]
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    fig.suptitle("Neural network from scratch — bundled 8×8 digits", fontsize=16, fontweight="bold")
    left.plot(loss[:, 0], loss[:, 1], color="#2563eb", linewidth=2)
    left.set(xlabel="Accepted conjugate-gradient iteration", ylabel="Regularized training loss",
             title="Training objective (log scale)", yscale="log")
    left.grid(alpha=0.2)
    right.imshow(confusion, cmap="Blues", vmin=0, vmax=max(1, confusion.max()))
    for i in range(10):
        for j in range(10):
            right.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=8,
                       color="white" if confusion[i, j] > confusion.max() / 2 else "#1e293b")
    right.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
              title=f"Held-out test: {metrics['metrics']['test']['accuracy']:.2%} accuracy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
