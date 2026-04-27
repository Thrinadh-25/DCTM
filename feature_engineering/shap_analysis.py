"""LightGBM-based SHAP feature importance."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.io import save_json
from utils.logger import get_logger

_log = get_logger("shap")


def compute_shap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sample_size: int = 50000,
    lgbm_params: dict | None = None,
    out_json: str = "configs/shap_scores.json",
    plot_path: str | None = "visualization/shap_importance.png",
    random_state: int = 42,
) -> dict:
    import lightgbm as lgb
    import shap

    rng = np.random.default_rng(random_state)
    n = len(X)
    if sample_size and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs, ys = X[idx], y[idx]
        _log.info(f"SHAP subsample: {n} -> {sample_size}")
    else:
        Xs, ys = X, y

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if lgbm_params:
        params.update(lgbm_params)

    n_classes = int(np.max(ys)) + 1
    objective = "binary" if n_classes <= 2 else "multiclass"
    if objective == "multiclass":
        params["num_class"] = n_classes
        params["objective"] = "multiclass"
    else:
        params["objective"] = "binary"

    _log.info(f"Training LightGBM ({objective}, n_classes={n_classes}) for SHAP ...")
    model = lgb.LGBMClassifier(**params)
    model.fit(Xs, ys)

    _log.info("Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    # For efficiency use a smaller sample for SHAP values
    shap_n = min(10000, len(Xs))
    Xs_small = Xs[:shap_n]
    sv = explainer.shap_values(Xs_small)
    if isinstance(sv, list):
        # multiclass: list per class — aggregate mean|shap| across classes
        abs_vals = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        abs_vals = np.abs(sv).mean(axis=0)
    # newer SHAP may return (samples, features, classes) -> collapse to 1D
    if abs_vals.ndim > 1:
        abs_vals = abs_vals.mean(axis=-1)

    ranked = sorted(zip(feature_names, abs_vals.tolist()), key=lambda t: t[1], reverse=True)
    result = {"ranked": [{"feature": n, "shap": s} for n, s in ranked]}
    save_json(result, out_json)
    _log.info(f"SHAP scores -> {out_json}")

    if plot_path:
        _plot_shap(ranked[:25], plot_path)
    return result


def _plot_shap(ranked_top, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n, _ in ranked_top][::-1]
        vals = [v for _, v in ranked_top][::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.25)))
        ax.barh(names, vals, color="darkorange")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title("Top Features by SHAP Importance")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        _log.warning(f"SHAP plot failed: {e}")
