"""One-hidden-layer digit classifier with explicit NumPy backpropagation.

Biases occupy the LAST column in each weight matrix, as in the coursework.
The output layer uses independent sigmoid units and summed binary cross entropy.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def initializeWeights(n_in, n_out, rng=None):
    """Xavier-style uniform initialization, including a bias column."""
    if n_in < 1 or n_out < 1:
        raise ValueError("Layer sizes must be positive.")
    rng = np.random.default_rng() if rng is None else rng
    epsilon = np.sqrt(6.0 / (n_in + n_out + 1))
    return rng.uniform(-epsilon, epsilon, size=(n_out, n_in + 1))


def sigmoid(z):
    """Stable sigmoid without clipping the underlying objective."""
    return expit(z)


def _matrix(data):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Expected a finite, two-dimensional feature matrix.")
    return data


def _bias(data):
    return np.column_stack((data, np.ones(data.shape[0])))


def nnObjFunction(params, *args):
    """Return regularized cross entropy and its analytic gradient.

    args: n_input, n_hidden, n_class, training_data, training_label, lambda.
    The L2 penalty is lambda / (2*N) times squared NON-bias weights.
    """
    n_input, n_hidden, n_class, data, labels, lambdaval = args
    if n_input < 1 or n_hidden < 1 or n_class < 2:
        raise ValueError("Invalid network dimensions.")
    if not np.isfinite(lambdaval) or lambdaval < 0:
        raise ValueError("Regularization must be finite and nonnegative.")
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if data.ndim != 2 or data.shape[1] != n_input or data.shape[0] == 0:
        raise ValueError("Training data does not match the network dimensions.")
    if labels.shape != (data.shape[0],) or not np.isfinite(labels).all():
        raise ValueError("Expected one finite label per sample.")
    if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels >= n_class)):
        raise ValueError("Labels must be integer class indices.")
    params = np.asarray(params, dtype=np.float64)
    cut = n_hidden * (n_input + 1)
    if params.shape != (cut + n_class * (n_hidden + 1),):
        raise ValueError("Parameter vector has the wrong shape.")
    w1 = params[:cut].reshape(n_hidden, n_input + 1)
    w2 = params[cut:].reshape(n_class, n_hidden + 1)
    x = _bias(data)
    hidden = sigmoid(x @ w1.T)
    hidden_bias = _bias(hidden)
    logits = hidden_bias @ w2.T
    if not np.isfinite(logits).all():
        raise ValueError("Network logits must be finite.")
    targets = np.eye(n_class)[labels.astype(int)]
    n = data.shape[0]
    # log(1 + exp(logit)) - target*logit avoids log(0) at saturated outputs.
    loss = np.sum(np.logaddexp(0.0, logits) - targets * logits) / n
    loss += lambdaval * (np.sum(w1[:, :-1] ** 2) + np.sum(w2[:, :-1] ** 2)) / (2 * n)
    output_error = sigmoid(logits) - targets
    hidden_error = (output_error @ w2[:, :-1]) * hidden * (1 - hidden)
    grad1 = hidden_error.T @ x / n
    grad2 = output_error.T @ hidden_bias / n
    grad1[:, :-1] += lambdaval * w1[:, :-1] / n
    grad2[:, :-1] += lambdaval * w2[:, :-1] / n
    return float(loss), np.concatenate((grad1.ravel(), grad2.ravel()))


def nnPredict(w1, w2, data):
    """Predict class indices from already-transformed features."""
    data = _matrix(data)
    w1, w2 = _matrix(w1), _matrix(w2)
    if w1.shape[1] != data.shape[1] + 1 or w2.shape[1] != w1.shape[0] + 1:
        raise ValueError("Weights and data have incompatible shapes.")
    if w1.shape[0] == 0 or w2.shape[0] < 2:
        raise ValueError("Invalid weight dimensions.")
    hidden = sigmoid(_bias(data) @ w1.T)
    # Sigmoid is monotonic; argmax of logits avoids saturated-probability ties.
    return np.argmax(_bias(hidden) @ w2.T, axis=1)


@dataclass
class FeatureTransform:
    """Training-only constant-feature mask and known dataset pixel scale."""

    mask: np.ndarray
    scale: float

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 1 or not self.mask.any():
            raise ValueError("At least one input feature must be selected.")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Pixel scale must be positive and finite.")

    @classmethod
    def fit(cls, training_data, scale):
        training_data = _matrix(training_data)
        if training_data.shape[0] == 0:
            raise ValueError("Cannot fit preprocessing on an empty training set.")
        return cls(np.std(training_data, axis=0) > 1e-5, float(scale))

    def apply(self, data):
        data = _matrix(data)
        if data.shape[1] != self.mask.size:
            raise ValueError("Raw feature count differs from the fitted transform.")
        return data[:, self.mask] / self.scale


@dataclass
class Model:
    w1: np.ndarray
    w2: np.ndarray
    transform: FeatureTransform

    def __post_init__(self):
        self.w1, self.w2 = _matrix(self.w1), _matrix(self.w2)
        # Validate saved dimensions even when there are no samples to predict.
        nnPredict(self.w1, self.w2, np.empty((0, int(self.transform.mask.sum()))))

    def predict(self, raw_data):
        return nnPredict(self.w1, self.w2, self.transform.apply(raw_data))

    def save(self, path):
        """Save arrays and preprocessing together, without pickle objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, version=np.array(1), w1=self.w1, w2=self.w2,
                                mask=self.transform.mask, scale=np.array(self.transform.scale))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if saved["version"].item() != 1:
                raise ValueError("Unsupported model format.")
            return cls(saved["w1"], saved["w2"],
                       FeatureTransform(saved["mask"], float(saved["scale"].item())))


def train_network(raw_data, labels, *, pixel_scale, n_hidden=32, n_class=10,
                  regularization=0.1, maxiter=150, seed=42):
    """Train with SciPy conjugate gradient; retain every accepted step's loss.

    Hyperparameters are supplied in advance. Validation/test data never enter
    preprocessing or optimization. The optimizer result records convergence.
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")
    transform = FeatureTransform.fit(raw_data, pixel_scale)
    data = transform.apply(raw_data)
    rng = np.random.default_rng(seed)
    w1 = initializeWeights(data.shape[1], n_hidden, rng)
    w2 = initializeWeights(n_hidden, n_class, rng)
    initial = np.concatenate((w1.ravel(), w2.ravel()))
    args = (data.shape[1], n_hidden, n_class, data, labels, regularization)
    history = [nnObjFunction(initial, *args)[0]]

    def record(params):
        history.append(nnObjFunction(params, *args)[0])

    result = minimize(nnObjFunction, initial, args=args, jac=True, method="CG",
                      callback=record, options={"maxiter": maxiter})
    cut = w1.size
    model = Model(result.x[:cut].reshape(w1.shape), result.x[cut:].reshape(w2.shape), transform)
    return model, result, history


if __name__ == "__main__":
    from train import main
    main()
