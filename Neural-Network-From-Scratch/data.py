"""Deterministic digit datasets; transformations are fitted by the model."""

from pathlib import Path
import hashlib

import numpy as np
from scipy.io import loadmat


def _blocks(mat, prefix):
    arrays, labels = [], []
    width = None
    for digit in range(10):
        key = f"{prefix}{digit}"
        if key not in mat:
            raise ValueError(f"Missing MAT dataset key: {key}")
        block = np.asarray(mat[key], dtype=np.float64)
        if block.ndim != 2 or block.shape[0] == 0 or block.shape[1] == 0:
            raise ValueError(f"Expected nonempty image rows in {key}.")
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 255)):
            raise ValueError(f"Expected finite 0..255 pixels in {key}.")
        if width is not None and block.shape[1] != width:
            raise ValueError("Digit blocks have inconsistent feature counts.")
        width = block.shape[1]
        arrays.append(block)
        labels.append(np.full(block.shape[0], digit, dtype=int))
    return np.concatenate(arrays), np.concatenate(labels)


def load_mnist(path, seed=42, validation_per_class=1000):
    """Read course-format train0..train9/test0..test9 arrays.

    Keep the supplied test partition untouched. Randomize each training class
    before reserving validation examples, then shuffle training/validation.
    """
    if validation_per_class < 1:
        raise ValueError("At least one validation example per class is required.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md.")
    mat = loadmat(path)
    X, y = _blocks(mat, "train")
    Xtest, ytest = _blocks(mat, "test")
    if X.shape[1] != Xtest.shape[1]:
        raise ValueError("Training and test feature counts differ.")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(y == digit))
        if indices.size <= validation_per_class:
            raise ValueError(f"Class {digit} has too few training examples.")
        validation.extend(indices[:validation_per_class])
        train.extend(indices[validation_per_class:])
    train = rng.permutation(train)
    validation = rng.permutation(validation)
    metadata = {"dataset": "mnist_all.mat (user supplied)", "pixel_scale": 255.0,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split": "per-class randomized validation; supplied test partition"}
    return (X[train], y[train]), (X[validation], y[validation]), (Xtest, ytest), metadata


def load_digits_demo(seed=42):
    """Bundled sklearn 8x8 digits, with disjoint stratified 60/20/20 splits.

    Per-class validation/test counts are rounded down. No network fetch.
    sklearn is used for loading data only, never for training the neural net.
    """
    try:
        from sklearn.datasets import load_digits
    except ImportError as error:
        raise ImportError("Install requirements-demo.txt for the digits demo.") from error
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(y == digit))
        n = indices.size // 5
        test.extend(indices[:n])
        validation.extend(indices[n:2*n])
        train.extend(indices[2*n:])
    train, validation, test = (rng.permutation(indices) for indices in (train, validation, test))
    metadata = {"dataset": "sklearn.datasets.load_digits (8x8, not MNIST)",
                "pixel_scale": 16.0, "split": "stratified 60/20/20; per-class counts rounded down",
                "data_sha256": hashlib.sha256(np.asarray(X, dtype="<f8").tobytes()
                                              + np.asarray(y, dtype="<i8").tobytes()).hexdigest(),
                "split_indices": {"train": train.tolist(), "validation": validation.tolist(),
                                  "test": test.tolist()}}
    return (X[train], y[train]), (X[validation], y[validation]), (X[test], y[test]), metadata
