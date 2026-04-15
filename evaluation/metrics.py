"""Shared metrics + plotting helpers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logger import get_logger

_log = get_logger("metrics")


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if y_proba is not None:
            classes = np.unique(y_true)
            if len(classes) == 2:
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    out["roc_auc"] = float("nan")
            else:
                out["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro",
                        labels=np.unique(y_true),
                    )
                )
        else:
            out["roc_auc"] = float("nan")
    except Exception as e:
        _log.warning(f"ROC-AUC failed: {e}")
        out["roc_auc"] = float("nan")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix: {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        # annotate cells only if small
        if cm.shape[0] <= 15:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"cm_{model_name}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as e:
        _log.warning(f"Confusion-matrix plot failed for {model_name}: {e}")
        return None


def compute_evasion_rate(y_true, y_pred, benign_label: int = 0) -> float:
    """ER = (#malicious samples predicted benign) / (#malicious samples)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    malicious = y_true != benign_label
    total = malicious.sum()
    if total == 0:
        return 0.0
    evaded = ((y_pred == benign_label) & malicious).sum()
    return float(evaded) / float(total)
