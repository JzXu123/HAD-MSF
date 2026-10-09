"""Numerical utilities used by the runner."""
import numpy as np


def rx(x: np.ndarray):
    cube = np.asarray(x, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("rx expects HxWxB")
    h, w, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    mean = pixels.mean(axis=0, keepdims=True)
    covariance = np.cov(pixels, rowvar=False)
    inverse = np.linalg.pinv(covariance + 1e-6 * np.eye(bands))
    scores = np.einsum("nd,de,ne->n", pixels - mean, inverse, pixels - mean)
    return scores.reshape(h, w)


class MinMaxNorm:
    def __init__(self, feature_range=(0, 1), eps=1e-12):
        self.feature_range = feature_range
        self.eps = eps
        self.min = None
        self.max = None

    def fit(self, x):
        self.min, self.max = x.min(), x.max()
        return self

    def transform(self, x):
        denominator = max(float(self.max - self.min), self.eps)
        x_std = (x - self.min) / denominator
        return x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def inverse_transform(self, x_norm):
        denominator = self.feature_range[1] - self.feature_range[0]
        x_std = (x_norm - self.feature_range[0]) / max(denominator, self.eps)
        return x_std * (self.max - self.min) + self.min


class ZScoreNorm:
    def __init__(self, eps=1e-6):
        self.means = None
        self.stds = None
        self.eps = eps

    def fit(self, x):
        self.means = np.mean(x, axis=(0, 1))
        self.stds = np.maximum(np.std(x, axis=(0, 1)), self.eps)
        return self

    def transform(self, x):
        if self.means is None:
            raise RuntimeError("call fit before transform")
        return (np.asarray(x, dtype=np.float32) - self.means) / self.stds

    def inverse_transform(self, x_norm):
        return np.asarray(x_norm) * self.stds + self.means
