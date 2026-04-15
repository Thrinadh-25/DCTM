"""Mutual-information feature ranking."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from utils.io import save_json
from utils.logger import get_logger

_log = get_logger("mi")


def compute_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    sample_size: int = 100000,
    random_state: int = 42,
    out_json: str = "configs/features_baseline.json",
    plot_path: str | None = "visualization/mi_importance.png",
) -> dict:
    """Ranks features by MI and writes top-k as JSON.

    Returns {"ranked": [(name, score), ...], "top_k": [...]}.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    if sample_size and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs, ys = X[idx], y[idx]
        _log.info(f"MI subsample: {n} -> {sample_size}")
    else:
        Xs, ys = X, y

    _log.info(f"Computing MI over {Xs.shape[1]} features ...")
    scores = mutual_info_classif(Xs, ys, random_state=random_state)
    ranked = sorted(zip(feature_names, scores.tolist()), key=lambda t: t[1], reverse=True)
    top = [name for name, _ in ranked[:top_k]]

    result = {
        "ranked": [{"feature": n, "mi": s} for n, s in ranked],
        "top_k": top,
        "k": top_k,
    }
    save_json(result, out_json)
    _log.info(f"MI top-{top_k} -> {out_json}")

    if plot_path:
        _plot_mi(ranked[:25], plot_path)
    return result


def _plot_mi(ranked_top: list[tuple[str, float]], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _ in ranked_top][::-1]
        vals = [v for _, v in ranked_top][::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.25)))
        ax.barh(names, vals, color="steelblue")
        ax.set_xlabel("Mutual Information")
        ax.set_title("Top Features by Mutual Information")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        _log.warning(f"MI plot failed: {e}")
