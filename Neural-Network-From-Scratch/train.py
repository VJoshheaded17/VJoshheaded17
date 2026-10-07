"""Reproducible training and evaluation entry point."""

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from data import load_digits_demo, load_mnist
from nnScript import Model, train_network


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "mnist"), default="digits")
    parser.add_argument("--data", type=Path, default=Path("data/mnist_all.mat"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.dataset == "digits":
        train, validation, test, metadata = load_digits_demo(args.seed)
    else:
        train, validation, test, metadata = load_mnist(args.data, args.seed)
    started = time.perf_counter()
    model, optimizer, history = train_network(
        *train, pixel_scale=metadata["pixel_scale"], n_hidden=args.hidden,
        regularization=args.regularization, maxiter=args.maxiter, seed=args.seed)
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.npz")
    predictions = {name: model.predict(split[0]) for name, split in
                   (("train", train), ("validation", validation), ("test", test))}
    # Exercise persisted preprocessing and weights, not just in-memory predictions.
    np.testing.assert_array_equal(Model.load(args.output / "model.npz").predict(test[0]),
                                  predictions["test"])
    metrics = {name: {"samples": len(split[1]),
                      "correct": int(np.sum(predictions[name] == split[1])),
                      "accuracy": float(np.mean(predictions[name] == split[1]))}
               for name, split in (("train", train), ("validation", validation), ("test", test))}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    if args.dataset == "digits":
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    report = {"dataset": metadata["dataset"], "seed": args.seed,
              "hyperparameters": {"hidden_units": args.hidden, "regularization": args.regularization,
                                  "maxiter": args.maxiter},
              "raw_features": train[0].shape[1],
              "selected_features": int(model.transform.mask.sum()), "metrics": metrics,
              "optimizer": {"method": "CG", "success": bool(optimizer.success),
                            "status": int(optimizer.status), "message": str(optimizer.message),
                            "iterations": int(optimizer.nit), "initial_loss": history[0],
                            "final_loss": float(optimizer.fun), "training_seconds": elapsed},
              "versions": versions,
              "evaluation_protocol": "Fixed parameters; train-only feature mask; test evaluated after training."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accepted_iteration", "regularized_training_loss"])
        writer.writerows(enumerate(history))
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (test[1], predictions["test"]), 1)
    with (args.output / "confusion.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_digit"] + [f"predicted_{i}" for i in range(10)])
        writer.writerows([i] + row.tolist() for i, row in enumerate(confusion))
    with (args.output / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index_in_test_split", "true_digit", "predicted_digit"])
        writer.writerows((i, int(true), int(pred)) for i, (true, pred) in
                         enumerate(zip(test[1], predictions["test"])))
    print(json.dumps(report, indent=2))
    print(f"Saved model, metrics, split provenance and predictions to {args.output}")
    return report


if __name__ == "__main__":
    main()
