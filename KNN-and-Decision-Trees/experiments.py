"""Paired repeated holdouts; KNN k selection uses inner validation only."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from data import load_car, load_wdbc
from decision import CategoricalDecisionTree
from knn import KNNClassifier


def split_indices(y, seed):
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)


def _stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _split_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def run_knn(X, y, repeats=20, seed=42, k_values=range(1, 52, 2)):
    k_values = sorted(set(k_values))
    rows, selected, splits = [], [], []
    for repeat in range(repeats):
        run_seed = seed + repeat
        train, test = split_indices(y, run_seed)
        fit, validation = train_test_split(train, test_size=0.2, stratify=y[train], random_state=run_seed + 100000)
        splits.append({"seed": run_seed, "train_sha256": _split_hash(train), "test_sha256": _split_hash(test),
                       "inner_fit_sha256": _split_hash(fit), "inner_validation_sha256": _split_hash(validation)})
        for variant, normalize in (("normalized", True), ("raw", False)):
            inner = KNNClassifier(normalize).fit(X[fit], y[fit])
            validation_predictions = inner.predict_many(X[validation], k_values)
            scores = [float(np.mean(validation_predictions[k] == y[validation])) for k in k_values]
            chosen = k_values[int(np.argmax(scores))]
            model = KNNClassifier(normalize).fit(X[train], y[train])
            resubstitution = model.predict_many(X[train], k_values)
            excluding_self = model.predict_many(X[train], k_values, exclude_indices=np.arange(len(train)))
            test_predictions = model.predict_many(X[test], k_values)
            for k in k_values:
                rows.append({"seed": run_seed, "variant": variant, "k": k,
                             "train_resubstitution": float(np.mean(resubstitution[k] == y[train])),
                             "train_excluding_self": float(np.mean(excluding_self[k] == y[train])),
                             "test_accuracy": float(np.mean(test_predictions[k] == y[test]))})
            selected.append({"seed": run_seed, "variant": variant, "selected_k": chosen,
                             "inner_validation_accuracy": max(scores), "test_samples": len(test),
                             "test_correct": int(np.sum(test_predictions[chosen] == y[test])),
                             "test_accuracy": float(np.mean(test_predictions[chosen] == y[test]))})
    curves = []
    for variant in ("normalized", "raw"):
        for k in k_values:
            group = [row for row in rows if row["variant"] == variant and row["k"] == k]
            result = {"variant": variant, "k": k}
            for metric in ("train_resubstitution", "train_excluding_self", "test_accuracy"):
                stats = _stats([row[metric] for row in group])
                result[f"{metric}_mean"], result[f"{metric}_std"] = stats["mean"], stats["std"]
            curves.append(result)
    summary = {}
    for variant in ("normalized", "raw"):
        group = [row for row in selected if row["variant"] == variant]
        summary[variant] = {"validation_selected_test_accuracy": _stats([row["test_accuracy"] for row in group]),
                            "selected_k_counts": dict(sorted(Counter(row["selected_k"] for row in group).items()))}
    return rows, selected, curves, summary, splits


def run_trees(X, y, repeats=100, seed=42):
    variants = {"entropy": ("entropy", None), "gini": ("gini", None), "entropy_purity85": ("entropy", 0.85)}
    rows, splits = [], []
    for repeat in range(repeats):
        run_seed = seed + repeat
        train, test = split_indices(y, run_seed)
        classes, counts = np.unique(y[train], return_counts=True)
        majority = classes[counts.argmax()]
        baseline = float(np.mean(y[test] == majority))
        splits.append({"seed": run_seed, "train_sha256": _split_hash(train), "test_sha256": _split_hash(test)})
        for variant, (criterion, purity) in variants.items():
            model = CategoricalDecisionTree(criterion, purity).fit(X[train], y[train])
            prediction = model.predict(X[test])
            rows.append({"seed": run_seed, "variant": variant,
                         "train_accuracy": float(np.mean(model.predict(X[train]) == y[train])),
                         "test_accuracy": float(np.mean(prediction == y[test])),
                         "test_samples": len(test), "test_correct": int(np.sum(prediction == y[test])),
                         "majority_baseline_accuracy": baseline, **model.statistics()})
    summary = {}
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        summary[variant] = {metric: _stats([row[metric] for row in group]) for metric in
                            ("train_accuracy", "test_accuracy", "nodes", "depth", "majority_baseline_accuracy")}
    return rows, summary, splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("knn", "tree", "all"))
    parser.add_argument("--wdbc", type=Path, help="Optional course-format WDBC CSV.")
    parser.add_argument("--car", type=Path, help="Optional course-format Car CSV.")
    parser.add_argument("--knn-repeats", type=int, default=20)
    parser.add_argument("--tree-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmarks"))
    args = parser.parse_args(argv)
    if args.knn_repeats < 1 or args.tree_repeats < 1:
        parser.error("Repeat counts must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "versions": {"python": platform.python_version(),
              "numpy": np.__version__, "scikit_learn": sklearn.__version__},
              "protocol": "Seeded stratified 80/20 holdouts; same partitions for paired variants; population std (ddof=0).",
              "caution": "Repeated holdouts overlap. Standard deviations describe split variation, not confidence intervals."}
    provenance = {}
    if args.experiment in ("knn", "all"):
        X, y, metadata = load_wdbc(args.wdbc)
        rows, selected, curves, summary, splits = run_knn(X, y, args.knn_repeats, args.seed)
        _write_csv(args.output / "knn_runs.csv", rows)
        _write_csv(args.output / "knn_selection.csv", selected)
        _write_csv(args.output / "knn_curves.csv", curves)
        report["knn"] = {"repeats": args.knn_repeats, "k_grid": list(range(1, 52, 2)), "summary": summary,
                         "selection": "Inner 80/20 split of outer training only; smallest k wins validation ties; refit on outer train."}
        provenance["knn"] = {"dataset": metadata, "splits": splits}
    if args.experiment in ("tree", "all"):
        X, y, metadata = load_car(args.car)
        rows, summary, splits = run_trees(X, y, args.tree_repeats, args.seed)
        _write_csv(args.output / "tree_runs.csv", rows)
        report["tree"] = {"repeats": args.tree_repeats, "summary": summary,
                          "settings": "Entropy, Gini, and entropy with purity threshold 0.85; settings fixed before evaluation."}
        provenance["tree"] = {"dataset": metadata, "splits": splits}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
