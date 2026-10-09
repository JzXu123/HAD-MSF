"""Supplementary evaluation protocols from the manuscript."""
import numpy as np


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo <= 1e-12 else (scores - lo) / (hi - lo)


def binary_metrics(scores, gt, threshold):
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(gt, dtype=bool).ravel()
    predicted = scores > float(threshold)
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall_pd": recall,
        "f1": f1,
        "pf": fp / max(fp + tn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def otsu_threshold(scores, bins=256):
    values = normalize_scores(scores).ravel()
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = histogram.astype(float) / max(histogram.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    total_mean = mean[-1]
    between = (total_mean * cumulative - mean) ** 2 / np.maximum(
        cumulative * (1.0 - cumulative), 1e-12
    )
    return float(centers[int(np.argmax(between))])


def validation_threshold(scores, gt, validation_fraction=0.2, seed=42):
    """Choose F1 on a 20% split and report on the held-out 80% split."""
    values = normalize_scores(scores).ravel()
    labels = np.asarray(gt, dtype=bool).ravel()
    if values.size != labels.size:
        raise ValueError("scores and gt must contain the same number of pixels")
    rng = np.random.default_rng(seed)
    order = rng.permutation(values.size)
    split = max(1, int(values.size * validation_fraction))
    validation, test = order[:split], order[split:]
    candidates = np.unique(np.quantile(values[validation], np.linspace(0, 1, 1001)))
    selected = max(candidates, key=lambda t: binary_metrics(
        values[validation], labels[validation], t
    )["f1"])
    result = binary_metrics(values[test], labels[test], selected)
    result["validation_fraction"] = float(validation_fraction)
    result["seed"] = int(seed)
    result["validation_threshold"] = float(selected)
    return result


def evaluate_operating_points(scores, gt, seed=42):
    normalized = normalize_scores(scores)
    otsu = otsu_threshold(normalized)
    result = {"otsu": binary_metrics(normalized, gt, otsu)}
    result["validation"] = validation_threshold(normalized, gt, seed=seed)
    return result


def add_gaussian_noise(cube, sigma, seed=42):
    """Add zero-mean noise at sigma times the cube-wide standard deviation."""
    data = np.asarray(cube, dtype=np.float32)
    rng = np.random.default_rng(seed)
    scale = max(float(data.std()), 1e-12)
    return data + rng.normal(0.0, float(sigma) * scale, data.shape).astype(np.float32)
