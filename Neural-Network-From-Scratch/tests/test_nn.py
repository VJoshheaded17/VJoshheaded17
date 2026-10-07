"""Numerical checks and data/model behavior; no MNIST download needed."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from data import load_digits_demo, load_mnist
from nnScript import FeatureTransform, Model, initializeWeights, nnObjFunction, nnPredict, train_network


class GradientTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_with_and_without_regularization(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(5, 3))
        labels = np.array([0, 1, 2, 1, 0])
        params = rng.normal(scale=0.3, size=2 * 4 + 3 * 3)
        for regularization in (0.0, 0.7):
            with self.subTest(regularization=regularization):
                args = (3, 2, 3, data, labels, regularization)
                _, analytic = nnObjFunction(params, *args)
                numerical = np.empty_like(params)
                for i in range(params.size):
                    plus, minus = params.copy(), params.copy()
                    plus[i] += 1e-6
                    minus[i] -= 1e-6
                    numerical[i] = (nnObjFunction(plus, *args)[0] - nnObjFunction(minus, *args)[0]) / 2e-6
                np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=2e-6)

    def test_bias_is_excluded_from_regularization(self):
        data = np.array([[0.0, 0.5], [1.0, 0.0]])
        w1 = np.array([[0.1, 0.2, 4.0], [0.3, 0.4, -5.0]])
        w2 = np.array([[0.2, 0.1, 7.0], [0.4, 0.3, -6.0]])
        params = np.concatenate((w1.ravel(), w2.ravel()))
        args = (2, 2, 2, data, np.array([0, 1]))
        loss0, grad0 = nnObjFunction(params, *args, 0.0)
        loss1, grad1 = nnObjFunction(params, *args, 0.8)
        expected = np.concatenate((np.column_stack((w1[:, :-1], np.zeros(2))).ravel(),
                                   np.column_stack((w2[:, :-1], np.zeros(2))).ravel()))
        np.testing.assert_allclose(grad1 - grad0, 0.8 * expected / 2)
        self.assertAlmostEqual(loss1 - loss0, 0.8 * np.sum(expected ** 2) / 4)

    def test_extreme_logits_have_finite_loss_and_gradient(self):
        w1 = np.array([[0.0, 0.0]])
        w2 = np.array([[0.0, 1000.0], [0.0, -1000.0]])
        loss, grad = nnObjFunction(np.concatenate((w1.ravel(), w2.ravel())),
                                   1, 1, 2, np.array([[0.0]]), np.array([1]), 0.0)
        self.assertEqual(loss, 2000.0)
        self.assertTrue(np.isfinite(grad).all())

    def test_invalid_labels_are_rejected(self):
        for labels in ([0.5], [-1], [2], [np.nan]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                nnObjFunction(np.zeros(6), 1, 1, 2, np.array([[0.0]]), labels, 0.0)


class ModelTests(unittest.TestCase):
    def test_known_weights_use_last_column_bias(self):
        pred = nnPredict(np.array([[10.0, -5.0]]), np.array([[-10.0, 5.0], [10.0, -5.0]]),
                         np.array([[0.0], [1.0]]))
        np.testing.assert_array_equal(pred, [0, 1])

    def test_mask_is_fitted_on_training_only(self):
        transform = FeatureTransform.fit(np.array([[0, 7, 10], [16, 7, 0]]), scale=16)
        np.testing.assert_array_equal(transform.mask, [True, False, True])
        validation = np.array([[8, 99, 4]])
        np.testing.assert_allclose(transform.apply(validation), [[0.5, 0.25]])

    def test_saved_model_retains_mask_scale_and_predictions(self):
        transform = FeatureTransform(np.array([True, False]), 16.0)
        model = Model(np.array([[10.0, -5.0]]), np.array([[-10.0, 5.0], [10.0, -5.0]]), transform)
        raw = np.array([[0, 3], [16, 99]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = Model.load(path)
            np.testing.assert_array_equal(restored.predict(raw), [0, 1])
            np.testing.assert_array_equal(restored.transform.mask, transform.mask)
            self.assertEqual(restored.transform.scale, 16.0)
            with np.load(path, allow_pickle=False) as saved:
                self.assertFalse(any(saved[key].dtype.hasobject for key in saved.files))

    def test_shape_and_constant_feature_errors(self):
        with self.assertRaises(ValueError):
            FeatureTransform.fit(np.ones((4, 2)), 16)
        with self.assertRaises(ValueError):
            FeatureTransform(np.array([True]), 0)
        with self.assertRaises(ValueError):
            Model(np.zeros((2, 4)), np.zeros((2, 3)), FeatureTransform(np.array([True]), 1))
        with self.assertRaises(ValueError):
            FeatureTransform(np.array([True]), 16).apply(np.zeros((2, 2)))

    def test_small_training_problem_learns_and_is_repeatable(self):
        data = np.array([[0, 0], [0, 0.1], [0.1, 0], [1, 1], [0.9, 1], [1, 0.9]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        options = dict(pixel_scale=1, n_hidden=3, n_class=2, regularization=0.01, maxiter=100, seed=5)
        model, result, history = train_network(data, labels, **options)
        self.assertLess(result.fun, history[0] / 4)
        np.testing.assert_array_equal(model.predict(data), labels)
        other, _, _ = train_network(data, labels, **options)
        np.testing.assert_array_equal(model.w1, other.w1)
        np.testing.assert_array_equal(model.w2, other.w2)

    def test_seeded_initialization(self):
        first = initializeWeights(4, 3, np.random.default_rng(12))
        second = initializeWeights(4, 3, np.random.default_rng(12))
        np.testing.assert_array_equal(first, second)


class DatasetTests(unittest.TestCase):
    def test_mnist_format_partitions_and_preserves_test_data(self):
        blocks = {}
        for digit in range(10):
            ids = np.arange(4 * digit, 4 * digit + 4)
            blocks[f"train{digit}"] = np.column_stack((ids, ids + 1)).astype(np.uint8)
            blocks[f"test{digit}"] = np.array([[100 + digit, 101 + digit]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist_all.mat"
            savemat(path, blocks)
            train, validation, test, _ = load_mnist(path, seed=10, validation_per_class=2)
            second = load_mnist(path, seed=10, validation_per_class=2)
            self.assertEqual(len(train[1]), 20)
            self.assertEqual(len(validation[1]), 20)
            self.assertFalse(set(train[0][:, 0]) & set(validation[0][:, 0]))
            self.assertEqual(set(train[0][:, 0]) | set(validation[0][:, 0]), set(range(40)))
            np.testing.assert_array_equal(test[0][:, 0], np.arange(100, 110))
            np.testing.assert_array_equal(test[1], np.arange(10))
            np.testing.assert_array_equal(train[0], second[0][0])
            blocks.pop("test5")
            savemat(path, blocks)
            with self.assertRaisesRegex(ValueError, "test5"):
                load_mnist(path, validation_per_class=2)

    def test_digits_splits_are_disjoint_and_reproducible(self):
        train, validation, test, metadata = load_digits_demo(seed=42)
        indices = metadata["split_indices"]
        train_ids, val_ids, test_ids = (set(indices[key]) for key in ("train", "validation", "test"))
        self.assertFalse(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        self.assertEqual(len(train_ids | val_ids | test_ids), 1797)
        self.assertEqual(len(train[1]) + len(validation[1]) + len(test[1]), 1797)
        self.assertTrue(all(set(split[1]) == set(range(10)) for split in (train, validation, test)))
        self.assertEqual(indices, load_digits_demo(seed=42)[3]["split_indices"])


if __name__ == "__main__":
    unittest.main()
