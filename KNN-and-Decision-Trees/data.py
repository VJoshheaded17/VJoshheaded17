"""Public benchmark and optional course-format CSV loaders."""

import csv
import hashlib
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def _fingerprint(X, y):
    return hashlib.sha256(np.asarray(X).tobytes() + np.asarray(y).tobytes()).hexdigest()


def load_wdbc(path=None):
    if path is None:
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
        source = "sklearn.datasets.load_breast_cancer (bundled WDBC)"
        label_names = {"0": "malignant", "1": "benign"}
    else:
        path = Path(path)
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or any(len(row) != 31 for row in rows):
            raise ValueError("WDBC CSV must have 30 features plus one label, without a header.")
        array = np.asarray(rows, dtype=float)
        X, labels = array[:, :30], array[:, 30]
        if not np.isfinite(array).all() or np.any((labels != 0) & (labels != 1)):
            raise ValueError("WDBC pixels/features must be finite; labels must be 0 or 1.")
        y = labels.astype(int)
        source = f"User-supplied course CSV: {path.name}"
        label_names = {"0": "course label 0", "1": "course label 1"}
    X, y = np.asarray(X, dtype="<f8"), np.asarray(y, dtype="<i8")
    return X, y, {"source": source, "samples": len(y), "features": X.shape[1],
                  "label_names": label_names, "data_sha256": _fingerprint(X, y)}


def load_car(path=None):
    path = DATA / "car.data" if path is None else Path(path)
    with path.open(newline="") as handle:
        rows = [[entry.strip() for entry in row] for row in csv.reader(handle)]
    if rows and rows[0] and rows[0][-1].lower() in {"class", "label", "quality", "target"}:
        rows = rows[1:]
    if not rows or any(len(row) != 7 or not all(row) for row in rows):
        raise ValueError("Car CSV must contain six categorical features and one nonempty label.")
    mapping = {"unacceptable": "unacc", "acceptable": "acc", "good": "good",
               "very_good": "vgood", "very good": "vgood", "verygood": "vgood",
               "unacc": "unacc", "acc": "acc", "vgood": "vgood"}
    if any(row[-1].lower() not in mapping for row in rows):
        raise ValueError("Unexpected Car Evaluation class name.")
    X = np.asarray([row[:6] for row in rows], dtype=str)
    y = np.asarray([mapping[row[-1].lower()] for row in rows], dtype=str)
    return X, y, {"source": "UCI Car Evaluation" if path == DATA / "car.data" else path.name,
                  "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "samples": len(y), "features": X.shape[1], "classes": sorted(set(y.tolist())),
                  "class_mapping": mapping}
