"""SMOTE oversampling — TRAINING split only."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE

from utils.logger import get_logger

_log = get_logger("smote")


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
    min_samples: int = 6,
    plot_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE on minority classes; classes with < min_samples are skipped.

    Returns oversampled X, y and logs the before/after class distribution.
    """
    before = Counter(y.tolist())
    _log.info(f"SMOTE before: {dict(before)}")

    valid_classes = [c for c, n in before.items() if n >= min_samples]
    if len(valid_classes) < 2:
        _log.warning("SMOTE skipped: fewer than 2 eligible classes")
        _maybe_plot(before, before, plot_path)
        return X, y

    mask = np.isin(y, valid_classes)
    X_v, y_v = X[mask], y[mask]

    k = min(k_neighbors, int(min(Counter(y_v.tolist()).values())) - 1)
    k = max(k, 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        X_res, y_res = sm.fit_resample(X_v, y_v)
    except Exception as e:
        _log.error(f"SMOTE failed ({e}); returning original data")
        _maybe_plot(before, before, plot_path)
        return X, y

    # Reattach any skipped classes unchanged
    skipped_mask = ~mask
    if skipped_mask.any():
        X_res = np.vstack([X_res, X[skipped_mask]])
        y_res = np.concatenate([y_res, y[skipped_mask]])

    after = Counter(y_res.tolist())
    _log.info(f"SMOTE after:  {dict(after)}")
    _maybe_plot(before, after, plot_path)
    return X_res, y_res


def _maybe_plot(before: Counter, after: Counter, path: str | None) -> None:
    if not path:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = sorted(set(list(before.keys()) + list(after.keys())))
        b = [before.get(c, 0) for c in classes]
        a = [after.get(c, 0) for c in classes]
        x = np.arange(len(classes))
        w = 0.4
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w / 2, b, w, label="Before SMOTE")
        ax.bar(x + w / 2, a, w, label="After SMOTE")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
        ax.set_ylabel("Samples")
        ax.set_title("Class Distribution Before/After SMOTE")
        ax.legend()
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        _log.info(f"Class distribution plot -> {path}")
    except Exception as e:
        _log.warning(f"Failed to save SMOTE plot: {e}")
