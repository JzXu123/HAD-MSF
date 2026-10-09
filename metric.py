"""ROC helpers with degenerate-label handling."""
import numpy as np
from sklearn import metrics


def roc_auc(dm: np.ndarray, gt: np.ndarray):
    scores = np.asarray(dm, dtype=float).ravel()
    labels = np.asarray(gt, dtype=int).ravel()
    if np.unique(labels).size < 2:
        return np.array([0., 1.]), np.array([0., 1.]), float("nan")
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    return fpr, tpr, float(metrics.auc(fpr, tpr))
