"""Label-free SeT+ training and all metrics reported in the paper."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from sklearn import metrics


def _normalize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi - lo <= 1e-12 else (scores - lo) / (hi - lo)


def calculate_all_metrics(dm: np.ndarray, gt: np.ndarray, n_thresholds=10001):
    """Return the three primary and five composite AUC measures.

    The composite definitions are the equations printed in the manuscript:
    TD=ROC+PdTau, BS=ROC-PfTau, ODP=ROC+PdTau-PfTau,
    TD-BS=PdTau-PfTau, SNPR=PdTau/PfTau.
    """
    scores = _normalize(dm).ravel()
    labels = np.asarray(gt, dtype=bool).ravel()
    if np.unique(labels).size < 2:
        raise ValueError("AUC metrics require both anomaly and background pixels")
    pf, pd, _ = metrics.roc_curve(labels, scores)
    auc_pd_pf = float(metrics.auc(pf, pd))
    thresholds = np.linspace(0.0, 1.0, int(n_thresholds))
    positives, negatives = labels, ~labels
    pd_values = np.asarray([(scores > t)[positives].sum() / positives.sum()
                            for t in thresholds])
    pf_values = np.asarray([(scores > t)[negatives].sum() / negatives.sum()
                            for t in thresholds])
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc_pd_tau = float(trapz(pd_values, thresholds))
    auc_pf_tau = float(trapz(pf_values, thresholds))
    values = {
        "auc_pd_pf": auc_pd_pf,
        "auc_pf_tau": auc_pf_tau,
        "auc_pd_tau": auc_pd_tau,
        "auc_td": auc_pd_pf + auc_pd_tau,
        "auc_bs": auc_pd_pf - auc_pf_tau,
        "auc_odp": auc_pd_pf + auc_pd_tau - auc_pf_tau,
        "auc_td_bs": auc_pd_tau - auc_pf_tau,
        "auc_snpr": auc_pd_tau / max(auc_pf_tau, 1e-12),
    }
    values["thresholds"] = thresholds
    values["pd_values"] = pd_values
    values["pf_values"] = pf_values
    return values


def train_model(x, model, criterion, cri_kwargs, epochs, optimizer, verbose=False):
    model.train()
    last_loss = None
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        value = criterion(x=x, y=y, **cri_kwargs)
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}: {value.item()}")
        value.backward()
        optimizer.step()
        last_loss = float(value.detach().cpu())
        if verbose:
            print(f"  epoch {epoch + 1:03d}/{epochs}: loss={last_loss:.6f}")
    return last_loss


def _save_best(path, dataset_name, model, x, gt, dm, loss_value, values):
    path.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        model_output = model(x).detach().cpu().numpy()
    metrics_only = {k: float(v) for k, v in values.items()
                    if np.isscalar(v) and k.startswith("auc_")}
    savemat(path / f"{dataset_name}_result.mat", {
        "model_output": model_output,
        "detection_map": dm,
        "ground_truth": gt,
        "train_loss": float(loss_value),
        "metrics": metrics_only,
        "thresholds": values["thresholds"],
        "pd_values": values["pd_values"],
        "pf_values": values["pf_values"],
    })
    (path / f"{dataset_name}_metrics.json").write_text(
        json.dumps(metrics_only, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
        normalized = _normalize(dm)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gt, cmap="gray")
        axes[0].set_title("Ground truth")
        axes[1].imshow(normalized, cmap="viridis")
        axes[1].set_title("Residual score")
        axes[2].plot(values["pf_values"], values["pd_values"])
        axes[2].set_xlabel("Pf")
        axes[2].set_ylabel("Pd")
        axes[2].set_title(f"ROC AUC={values['auc_pd_pf']:.4f}")
        axes[3].plot(values["thresholds"], values["pd_values"], label="Pd")
        axes[3].plot(values["thresholds"], values["pf_values"], label="Pf")
        axes[3].set_xlabel("threshold")
        axes[3].legend()
        fig.tight_layout()
        fig.savefig(path / f"{dataset_name}_curves.png", dpi=220)
        plt.close(fig)
    except ImportError:
        pass


def separation_training(
    x, gt, model, loss, mask, optimizer, epochs=30, output_iter=1,
    max_iter=1, verbose=True, dataset_name="scene", result_root="results"
):
    """Train without labels and save the best residual plus paper metrics."""
    from .detect import detect
    best_loss = float("inf")
    best_dm = None
    best_values = {}
    history = []
    save_path = Path(result_root) / model.name
    for iteration in range(1, int(max_iter) + 1):
        if verbose:
            print(f"iteration {iteration}/{max_iter}")
        train_loss = train_model(
            x, model, loss, {"mask": mask}, epochs, optimizer, verbose
        )
        history.append(train_loss)
        model.eval()
        with torch.no_grad():
            prediction = model(x)
            dm = detect(x, prediction)
        mask.update(dm)
        np_dm = dm.detach().cpu().numpy()
        values = calculate_all_metrics(np_dm, gt)
        if verbose:
            print(f"  candidate: loss={train_loss:.6f}, "
                  f"AUC(Pd,Pf)={values['auc_pd_pf']:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            best_dm = np_dm.copy()
            best_values = values
            _save_best(save_path, dataset_name, model, x, gt, best_dm,
                       best_loss, values)
    return best_dm, history, best_values
