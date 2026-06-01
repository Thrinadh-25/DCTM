"""Clean vs adversarial evaluation for every trained IDS model."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.classical import CLASSICAL_MODELS
from models.deep_learning import DEEP_MODELS
from utils.logger import get_logger

from .metrics import compute_evasion_rate, compute_metrics

_log = get_logger("evasion")


def _load(name: str, feature_set: str, models_dir: str):
    is_deep = name in DEEP_MODELS
    ext = "pt" if is_deep else "pkl"
    path = os.path.join(models_dir, f"{name}_{feature_set}.{ext}")
    if not os.path.exists(path):
        return None
    if is_deep:
        return DEEP_MODELS[name].load(path)
    return CLASSICAL_MODELS[name].load(path)


def evaluate_evasion(
    X_test_clean: np.ndarray,
    y_test_clean: np.ndarray,
    X_adv: np.ndarray,
    y_adv: np.ndarray,
    cfg: dict,
    feature_set: str,
    benign_label: int = 0,
    model_names: list[str] | None = None,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """For each model: evaluate on clean and on adversarial data; report drops."""
    models_dir = cfg["paths"]["models"]
    all_names = list(CLASSICAL_MODELS.keys()) + list(DEEP_MODELS.keys())
    names = model_names or all_names

    rows = []
    print(f"\n[Evaluate] Evasion test on {len(names)} models  |  feature_set={feature_set}")
    for name in tqdm(names, desc="Evaluating", unit="model", ncols=90):
        try:
            mdl = _load(name, feature_set, models_dir)
            if mdl is None:
                tqdm.write(f"  [SKIP] {name} — no checkpoint")
                _log.warning(f"[{name}] no checkpoint for {feature_set}, skipping")
                continue
            yp_clean = mdl.predict(X_test_clean)
            try:
                prob_clean = mdl.predict_proba(X_test_clean)
            except Exception:
                prob_clean = None
            mc = compute_metrics(y_test_clean, yp_clean, prob_clean)
            er_clean = compute_evasion_rate(y_test_clean, yp_clean, benign_label)

            yp_adv = mdl.predict(X_adv)
            try:
                prob_adv = mdl.predict_proba(X_adv)
            except Exception:
                prob_adv = None
            ma = compute_metrics(y_adv, yp_adv, prob_adv)
            er_adv = compute_evasion_rate(y_adv, yp_adv, benign_label)

            rows.append({
                "model": name,
                "feature_set": feature_set,
                "clean_accuracy": mc["accuracy"],
                "clean_f1": mc["f1"],
                "clean_er": round(er_clean, 4),
                "adv_accuracy": ma["accuracy"],
                "adv_f1": ma["f1"],
                "adv_er": round(er_adv, 4),
                "accuracy_drop": round(mc["accuracy"] - ma["accuracy"], 4),
                "f1_drop": round(mc["f1"] - ma["f1"], 4),
            })
            tqdm.write(
                f"  {name:<22}  clean_ER={er_clean:.3f}  adv_ER={er_adv:.3f}  "
                f"F1_drop={mc['f1'] - ma['f1']:+.3f}"
            )
            _log.info(
                f"[{name}] clean ER={er_clean:.3f} -> adv ER={er_adv:.3f} | "
                f"F1 {mc['f1']:.3f} -> {ma['f1']:.3f}"
            )
        except Exception as e:
            tqdm.write(f"  [ERROR] {name}: {e}")
            _log.exception(f"Eval failed for {name}: {e}")

    df = pd.DataFrame(rows)
    if out_csv and len(df):
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        _log.info(f"Evasion results -> {out_csv}")
        _plot_evasion(df, cfg)
    return df


def _plot_evasion(df: pd.DataFrame, cfg: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz_dir = cfg["paths"]["visualization"]
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
        ref = cfg["evaluation"].get("demgan_reference_er", 0.9742)

        names = df["model"].tolist()
        er = df["adv_er"].tolist()
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, er, color="crimson")
        ax.axhline(ref, color="black", linestyle="--", label=f"DEMGAN ref {ref:.4f}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Evasion Rate")
        ax.set_title("Adversarial Evasion Rate per IDS Model")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(viz_dir, "evasion_comparison.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        _log.info(f"Evasion plot -> {path}")
    except Exception as e:
        _log.warning(f"Evasion plot failed: {e}")
