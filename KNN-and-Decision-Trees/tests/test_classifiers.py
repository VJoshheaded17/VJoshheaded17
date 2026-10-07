"""Regression checks for normalization, voting, tree fallback and evaluation."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from data import load_car, load_wdbc
from decision import CategoricalDecisionTree, impurity, information_gain
from experiments import run_knn, run_trees, split_indices
from knn import KNNClassifier, MinMaxScaler


class KNNTests(unittest.TestCase):
    def test_training_extrema_are_reused_for_queries(self):
        scaler = MinMaxScaler.fit(np.array([[10, 7], [20, 7]]))
        np.testing.assert_allclose(scaler.transform([[15, 7], [30, 8]]), [[0.5, 0], [2, 1]])
        # Held-out values may exceed [0,1]; they must not be rescaled separately.
        self.assertTrue(np.isfinite(scaler.transform([[15, 7]])).all())

    def test_multiclass_votes_and_batched_predictions(self):
        model = KNNClassifier(False).fit([[0], [2], [4], [8]], ["a", "b", "b", "c"])
        predictions = model.predict_many([[1.8], [7.9]], [1, 3])
        np.testing.assert_array_equal(predictions[1], ["b", "c"])
        np.testing.assert_array_equal(predictions[3], ["b", "b"])
        np.testing.assert_array_equal(model.predict([[1.8], [7.9]], 3), predictions[3])

    def test_distance_and_vote_ties_are_deterministic(self):
        model = KNNClassifier(False).fit([[0], [2]], ["z", "a"])
        self.assertEqual(model.predict([[1]], 1)[0], "z")
        self.assertEqual(model.predict([[1]], 2)[0], "a")

    def test_self_exclusion_is_by_index_preserving_duplicate_neighbors(self):
        X = np.array([[0.0], [0.0], [10.0]])
        model = KNNClassifier(False).fit(X, [0, 1, 1])
        np.testing.assert_array_equal(model.predict(X, 1, exclude_indices=np.arange(3)), [1, 0, 0])
        with self.assertRaises(ValueError):
            model.predict(X, 3, exclude_indices=np.arange(3))

    def test_invalid_data_k_and_query_shapes(self):
        with self.assertRaises(ValueError):
            KNNClassifier().fit([[np.nan]], [0])
        model = KNNClassifier().fit([[0], [1]], [0, 1])
        for k in (0, 3, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                model.predict([[0]], k)
        with self.assertRaises(ValueError):
            model.predict([[0, 1]], 1)


class TreeTests(unittest.TestCase):
    def test_entropy_and_gini_known_values(self):
        self.assertEqual(impurity(["a", "a", "b", "b"]), 1.0)
        self.assertEqual(impurity(["a", "a", "b", "b"], "gini"), 0.5)
        self.assertEqual(impurity([]), 0.0)
        X = np.array([["red"], ["red"], ["blue"], ["blue"]])
        y = np.array(["a", "a", "b", "b"])
        self.assertEqual(information_gain(0, y, X), 1.0)
        self.assertEqual(information_gain(0, y, X, "gini"), 0.5)

    def test_exhausted_attributes_return_actual_majority_class(self):
        for criterion in ("entropy", "gini"):
            model = CategoricalDecisionTree(criterion).fit([["same"], ["same"], ["same"]], ["b", "a", "b"])
            np.testing.assert_array_equal(model.predict([["same"], ["unseen"]]), ["b", "b"])

    def test_unseen_values_use_local_node_majority(self):
        X = [["red", "small"], ["red", "large"], ["blue", "small"], ["blue", "large"]]
        y = ["a", "a", "b", "c"]
        model = CategoricalDecisionTree().fit(X, y)
        np.testing.assert_array_equal(model.predict([["blue", "unknown"], ["purple", "small"]]), ["b", "a"])

    def test_zero_gain_interactions_can_be_learned(self):
        X = [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]]
        y = ["no", "yes", "yes", "no"]
        for criterion in ("entropy", "gini"):
            np.testing.assert_array_equal(CategoricalDecisionTree(criterion).fit(X, y).predict(X), y)

    def test_purity_and_depth_stopping(self):
        X, y = [["a"]] * 17 + [["b"]] * 3, ["major"] * 17 + ["minor"] * 3
        full = CategoricalDecisionTree().fit(X, y)
        early = CategoricalDecisionTree(purity_threshold=0.85).fit(X, y)
        depth = CategoricalDecisionTree(max_depth=0).fit(X, y)
        self.assertEqual(full.statistics()["nodes"], 3)
        self.assertEqual(early.statistics()["nodes"], 1)
        self.assertEqual(depth.statistics()["depth"], 0)
        self.assertTrue(all(early.predict(X) == "major"))

    def test_invalid_tree_input_is_rejected(self):
        with self.assertRaises(ValueError):
            CategoricalDecisionTree("wrong")
        with self.assertRaises(ValueError):
            CategoricalDecisionTree(purity_threshold=0.4)
        with self.assertRaises(ValueError):
            CategoricalDecisionTree().fit([], [])
        with self.assertRaises(ValueError):
            CategoricalDecisionTree().fit([["a"]], ["a"]).predict([["a", "b"]])


class ExperimentTests(unittest.TestCase):
    def test_split_is_seeded_and_disjoint(self):
        y = np.array([0] * 30 + [1] * 30)
        train, test = split_indices(y, 42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(60)))
        np.testing.assert_array_equal(train, split_indices(y, 42)[0])

    def test_paired_knn_results_and_selection_repeat_exactly(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(100, 2))
        y = (X[:, 0] > 0).astype(int)
        first = run_knn(X, y, repeats=2, seed=7, k_values=[1, 3, 5])
        self.assertEqual(first, run_knn(X, y, repeats=2, seed=7, k_values=[1, 3, 5]))
        self.assertEqual(len(first[0]), 12)
        self.assertEqual(len(first[1]), 4)
        for selection in first[1]:
            row = next(r for r in first[0] if r['seed'] == selection['seed']
                       and r['variant'] == selection['variant'] and r['k'] == selection['selected_k'])
            self.assertEqual(row['test_accuracy'], selection['test_accuracy'])
        self.assertTrue(all(r['train_resubstitution'] == 1 for r in first[0] if r['k'] == 1))

    def test_paired_tree_variants_and_repeatability(self):
        X = np.array([["red", "small"], ["red", "big"], ["blue", "small"], ["blue", "big"]] * 10)
        y = np.array(["a", "a", "b", "c"] * 10)
        first = run_trees(X, y, repeats=2, seed=9)
        self.assertEqual(first, run_trees(X, y, repeats=2, seed=9))
        self.assertEqual(len(first[0]), 6)
        self.assertTrue(all(r['test_accuracy'] == 1 for r in first[0]))

    def test_loaders_and_optional_course_headers(self):
        X, y, _ = load_wdbc()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(set(y), {0, 1})
        X, y, _ = load_car()
        self.assertEqual(X.shape, (1728, 6))
        self.assertEqual(set(y), {"unacc", "acc", "good", "vgood"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "car.csv"
            path.write_text("buying,maint,doors,persons,luggage,safety,label\nlow,low,2,4,small,high,very_good\n")
            self.assertEqual(load_car(path)[1][0], "vgood")
            path.write_text("\n")
            with self.assertRaises(ValueError):
                load_car(path)


if __name__ == "__main__":
    unittest.main()
