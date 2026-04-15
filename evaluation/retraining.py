"""Adversarial retraining: augment train with adv samples, retrain, re-evaluate."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models.trainer import train_all_models
from utils.logger import get_logger

from .evasion_evaluator import evaluate_evasion

_log = get_logger("retrain")


def adversarial_retrain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_adv: np.ndarray,
    y_adv: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: dict,
    feature_set: str,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """Concatenate adv samples into the training set, retrain all models, re-evaluate."""
    _log.info(
        f"Augmenting train ({len(X_train)}) + adv ({len(X_adv)}) -> {len(X_train) + len(X_adv)}"
    )
    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_adv], axis=0)

    # Use a separate feature-set tag so we don't overwrite the pre-adv checkpoints
    retrain_tag = f"{feature_set}_retrained"
    _log.info(f"Retraining all models under tag '{retrain_tag}'")
    train_all_models(
        X_aug, y_aug, X_test, y_test,
        num_classes=num_classes,
        cfg=cfg,
        feature_set=retrain_tag,
    )

    df = evaluate_evasion(
        X_test_clean=X_test, y_test_clean=y_test,
        X_adv=X_adv, y_adv=y_adv,
        cfg=cfg, feature_set=retrain_tag,
        out_csv=out_csv,
    )
    if out_csv and len(df):
        _plot_retrain(df, cfg, feature_set, retrain_tag)
    return df


def _plot_retrain(df: pd.DataFrame, cfg: dict, baseline_tag: str, retrain_tag: str) -> None:
    """Overlay baseline vs retrained ER/F1 — requires baseline evasion CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results_dir = cfg["paths"]["results"]
        base_csv = os.path.join(results_dir, f"evasion_{baseline_tag}.csv")
        if not os.path.exists(base_csv):
            return
        base = pd.read_csv(base_csv)
        merged = base.merge(df, on="model", suffixes=("_base", "_retrain"))
        names = merged["model"].tolist()
        x = np.arange(len(names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w / 2, merged["adv_er_base"], w, label="Before retrain", color="crimson")
        ax.bar(x + w / 2, merged["adv_er_retrain"], w, label="After retrain", color="seagreen")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Adversarial Evasion Rate")
        ax.set_title("Retraining Impact on Evasion Rate")
        ax.set_ylim(0, 1.05); ax.legend()
        fig.tight_layout()
        viz = cfg["paths"]["visualization"]
        Path(viz).mkdir(parents=True, exist_ok=True)
        out = os.path.join(viz, "retraining_improvement.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        _log.info(f"Retraining plot -> {out}")
    except Exception as e:
        _log.warning(f"Retrain plot failed: {e}")
