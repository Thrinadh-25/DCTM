"""Hybrid (MI + SHAP) feature selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.io import load_json, save_json
from utils.logger import get_logger

_log = get_logger("hybrid")


def _normalize(values: list[float]) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return ((v - lo) / (hi - lo)).tolist()


def select_hybrid_features(
    mi_path: str = "configs/features_baseline.json",
    shap_path: str = "configs/shap_scores.json",
    top_k: int = 20,
    weight_mi: float = 0.5,
    weight_shap: float = 0.5,
    out_json: str = "configs/features_dctm.json",
    plot_path: str | None = "visualization/feature_importance_comparison.png",
) -> dict:
    mi = load_json(mi_path)
    sh = load_json(shap_path)
    mi_map = {r["feature"]: r["mi"] for r in mi["ranked"]}
    sh_map = {r["feature"]: r["shap"] for r in sh["ranked"]}

    feats = sorted(set(mi_map) | set(sh_map))
    mi_vals = [mi_map.get(f, 0.0) for f in feats]
    sh_vals = [sh_map.get(f, 0.0) for f in feats]
    mi_n = _normalize(mi_vals)
    sh_n = _normalize(sh_vals)
    hybrid = [weight_mi * m + weight_shap * s for m, s in zip(mi_n, sh_n)]

    ranked = sorted(zip(feats, hybrid, mi_n, sh_n), key=lambda t: t[1], reverse=True)
    top = [name for name, *_ in ranked[:top_k]]

    result = {
        "ranked": [
            {"feature": f, "hybrid": h, "mi_norm": m, "shap_norm": s}
            for f, h, m, s in ranked
        ],
        "top_k": top,
        "k": top_k,
        "weights": {"mi": weight_mi, "shap": weight_shap},
    }
    save_json(result, out_json)
    _log.info(f"Hybrid top-{top_k} -> {out_json}")

    if plot_path:
        _plot_comparison(ranked[:top_k], plot_path)
    return result


def _plot_comparison(ranked_top, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [f for f, *_ in ranked_top][::-1]
        mi = [m for _, _, m, _ in ranked_top][::-1]
        sh = [s for _, _, _, s in ranked_top][::-1]
        hy = [h for _, h, _, _ in ranked_top][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(names) * 0.25)), sharey=True)
        axes[0].barh(names, mi, color="steelblue"); axes[0].set_title("MI (norm)")
        axes[1].barh(names, sh, color="darkorange"); axes[1].set_title("SHAP (norm)")
        axes[2].barh(names, hy, color="seagreen"); axes[2].set_title("Hybrid")
        for ax in axes:
            ax.set_xlim(0, 1.05)
        fig.suptitle("Feature Importance: MI vs SHAP vs Hybrid")
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        _log.warning(f"Hybrid comparison plot failed: {e}")
