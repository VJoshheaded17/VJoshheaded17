"""Euclidean KNN and training-fitted min-max scaling, implemented in NumPy."""

from dataclasses import dataclass
import numpy as np


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("Expected a finite two-dimensional feature matrix.")
    return X


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, X):
        X = _features(X)
        if len(X) == 0:
            raise ValueError("Cannot fit on empty data.")
        minimum = X.min(axis=0)
        span = X.max(axis=0) - minimum
        # A constant training feature needs a finite denominator.
        span[span == 0] = 1.0
        return cls(minimum, span)

    def transform(self, X):
        X = _features(X)
        if X.shape[1] != len(self.minimum):
            raise ValueError("Feature count differs from fitted data.")
        return (X - self.minimum) / self.span


class KNNClassifier:
    """Deterministic uniform voting, with batched distance evaluation.

    Distance ties retain training-row order; vote ties choose the smallest
    sorted class. Excluding a training row is explicit by index, never by value.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y):
        X = _features(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or len(X) == 0:
            raise ValueError("Expected a nonempty training set with one label per row.")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("Numeric labels must be finite.")
        self.scaler_ = MinMaxScaler.fit(X) if self.normalize else None
        self.X_ = (self.scaler_.transform(X) if self.normalize else X).copy()
        self.classes_, self.encoded_ = np.unique(y, return_inverse=True)
        return self

    def predict_many(self, X, k_values, *, exclude_indices=None):
        if not hasattr(self, "X_"):
            raise ValueError("Fit the classifier before predicting.")
        X = _features(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("Feature count differs from fitted data.")
        X = self.scaler_.transform(X) if self.normalize else X
        k_values = list(k_values)
        if not k_values or any(not isinstance(k, (int, np.integer)) or isinstance(k, bool) for k in k_values):
            raise ValueError("k values must be integers.")
        k_values = sorted(set(int(k) for k in k_values))
        limit = len(self.X_)
        if exclude_indices is not None:
            exclude_indices = np.asarray(exclude_indices)
            if exclude_indices.shape != (len(X),) or not np.issubdtype(exclude_indices.dtype, np.integer):
                raise ValueError("Provide one integer exclusion index per query.")
            if np.any((exclude_indices < 0) | (exclude_indices >= limit)):
                raise ValueError("Exclusion indices are outside the training set.")
            limit -= 1
        if k_values[0] < 1 or k_values[-1] > limit:
            raise ValueError(f"k must lie between 1 and {limit} available neighbors.")
        predictions = {k: np.empty(len(X), dtype=self.classes_.dtype) for k in k_values}
        requested = set(k_values)
        for start in range(0, len(X), 64):
            queries = X[start:start + 64]
            distances = np.sum((queries[:, None, :] - self.X_[None, :, :]) ** 2, axis=2)
            if not np.isfinite(distances).all():
                raise ValueError("Distances overflowed; rescale input features.")
            if exclude_indices is not None:
                distances[np.arange(len(queries)), exclude_indices[start:start + len(queries)]] = np.inf
            order = np.argsort(distances, axis=1, kind="stable")[:, :k_values[-1]]
            neighbors = self.encoded_[order]
            counts = np.zeros((len(queries), len(self.classes_)), dtype=int)
            for i in range(k_values[-1]):
                counts[np.arange(len(queries)), neighbors[:, i]] += 1
                if i + 1 in requested:
                    predictions[i + 1][start:start + len(queries)] = self.classes_[counts.argmax(axis=1)]
        return predictions

    def predict(self, X, k=3, *, exclude_indices=None):
        return self.predict_many(X, [k], exclude_indices=exclude_indices)[k]


if __name__ == "__main__":
    import sys
    from experiments import main
    main(["knn"] + sys.argv[1:])
