"""Multiway categorical decision trees with entropy or Gini splitting."""

import numpy as np


def impurity(y, criterion="entropy"):
    if criterion not in ("entropy", "gini"):
        raise ValueError("criterion must be entropy or gini.")
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    if criterion == "entropy":
        return float(-np.sum(probabilities * np.log2(probabilities)))
    return float(1 - np.sum(probabilities ** 2))


def information_gain(attribute, y, data, criterion="entropy"):
    data, y = np.asarray(data), np.asarray(y)
    if data.ndim != 2 or y.ndim != 1 or len(data) != len(y) or len(y) == 0:
        raise ValueError("Expected nonempty rows and aligned labels.")
    if attribute < 0 or attribute >= data.shape[1]:
        raise ValueError("Attribute index is outside the feature matrix.")
    score = impurity(y, criterion)
    for value in np.unique(data[:, attribute]):
        part = y[data[:, attribute] == value]
        score -= len(part) / len(y) * impurity(part, criterion)
    return float(score)


class CategoricalDecisionTree:
    """Each node retains its majority class for unseen-category fallback.

    Class ties use sorted order; split ties use the smallest feature index.
    Zero-gain splits remain allowed so interactions such as XOR can be learned.
    """

    def __init__(self, criterion="entropy", purity_threshold=None, max_depth=None):
        if criterion not in ("entropy", "gini"):
            raise ValueError("criterion must be entropy or gini.")
        if purity_threshold is not None and not (0.5 <= purity_threshold <= 1):
            raise ValueError("purity_threshold must lie between 0.5 and 1.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a nonnegative integer.")
        self.criterion = criterion
        self.purity_threshold = purity_threshold
        self.max_depth = max_depth

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=str), np.asarray(y, dtype=str)
        if X.ndim != 2 or X.shape[1] == 0 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("Expected a nonempty categorical matrix and aligned labels.")
        self.n_features_ = X.shape[1]
        self.tree_ = self._build(X, y, list(range(X.shape[1])), depth=0)
        return self

    def _build(self, X, y, attributes, depth):
        classes, counts = np.unique(y, return_counts=True)
        majority = str(classes[counts.argmax()])
        node = {"prediction": majority, "samples": int(len(y))}
        if (len(classes) == 1 or not attributes
                or (self.max_depth is not None and depth >= self.max_depth)
                or (self.purity_threshold is not None and counts.max() / len(y) >= self.purity_threshold)):
            return node
        gains = [information_gain(i, y, X, self.criterion) for i in attributes]
        attribute = attributes[int(np.argmax(gains))]
        node["attribute"] = attribute
        node["branches"] = {}
        remaining = [i for i in attributes if i != attribute]
        for value in np.unique(X[:, attribute]):
            selected = X[:, attribute] == value
            node["branches"][str(value)] = self._build(X[selected], y[selected], remaining, depth + 1)
        return node

    def predict(self, X):
        if not hasattr(self, "tree_"):
            raise ValueError("Fit the tree before predicting.")
        X = np.asarray(X, dtype=str)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Feature count differs from fitted data.")
        predictions = []
        for row in X:
            node = self.tree_
            while "attribute" in node:
                child = node["branches"].get(str(row[node["attribute"]]))
                if child is None:
                    break
                node = child
            predictions.append(node["prediction"])
        return np.asarray(predictions, dtype=str)

    def statistics(self):
        if not hasattr(self, "tree_"):
            raise ValueError("Fit the tree before requesting statistics.")
        def visit(node, depth):
            children = [visit(child, depth + 1) for child in node.get("branches", {}).values()]
            return (1 + sum(c[0] for c in children), max([depth] + [c[1] for c in children]))
        nodes, depth = visit(self.tree_, 0)
        return {"nodes": nodes, "depth": depth}


if __name__ == "__main__":
    import sys
    from experiments import main
    main(["tree"] + sys.argv[1:])
