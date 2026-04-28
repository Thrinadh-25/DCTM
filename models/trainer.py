"""Unified training loop for all 10 IDS models (6 classical + 4 deep).

Saves each model to data/models/{name}_{feature_set}.{pkl|pt}
and returns a metrics dict. This is model-agnostic: every model exposes the
same .train / .predict / .predict_proba / .save / .load interface.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from models.classical import CLASSICAL_MODELS
from models.deep_learning import DEEP_MODELS
from utils.logger import get_logger

_log = get_logger("trainer")


def _classical_ctor(name: str, cfg_cls: dict):
    params = dict(cfg_cls.get(name, {}))
    Cls = CLASSICAL_MODELS[name]
    return Cls(**params)


def _deep_ctor(name: str, input_dim: int, num_classes: int, cfg_dl: dict):
    top = dict(cfg_dl)
    per_model = top.pop(name, {}) or {}
    common = {
        "lr": top.get("lr", 1e-3),
        "weight_decay": top.get("weight_decay", 1e-4),
        "batch_size": top.get("batch_size", 512),
        "epochs": top.get("epochs", 30),
        "patience": top.get("patience", 10),
    }
    Cls = DEEP_MODELS[name]
    return Cls(input_dim=input_dim, num_classes=num_classes, **common, **per_model)


def _model_path(models_dir: str, name: str, feature_set: str, is_deep: bool) -> str:
    ext = "pt" if is_deep else "pkl"
    return os.path.join(models_dir, f"{name}_{feature_set}.{ext}")


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: dict,
    feature_set: str,
    include: list[str] | None = None,
) -> dict:
    """Train every model. Returns per-model dict with metrics and saved paths."""
    from evaluation.metrics import compute_metrics, plot_confusion_matrix

    models_dir = cfg["paths"]["models"]
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    cm_dir = os.path.join(cfg["paths"]["visualization"], "confusion_matrices")

    classical_cfg = cfg.get("classical", {})
    deep_cfg = cfg.get("deep_learning", {})

    results: dict[str, dict] = {}
    input_dim = X_train.shape[1]

    all_names = list(CLASSICAL_MODELS.keys()) + list(DEEP_MODELS.keys())
    names = include or all_names

    for name in names:
        try:
            is_deep = name in DEEP_MODELS
            path = _model_path(models_dir, name, feature_set, is_deep)

            if os.path.exists(path):
                _log.info(f"=== Skipping [{name}] ({feature_set}) — checkpoint exists at {path} ===")
                if is_deep:
                    Cls = DEEP_MODELS[name]
                    model = Cls.load(path)
                else:
                    Cls = CLASSICAL_MODELS[name]
                    model = Cls.load(path)
                train_time = 0.0
            else:
                t0 = time.time()
                if is_deep:
                    model = _deep_ctor(name, input_dim, num_classes, deep_cfg)
                else:
                    model = _classical_ctor(name, classical_cfg)
                _log.info(f"=== Training [{name}] ({feature_set}) ===")
                model.train(X_train, y_train)
                train_time = time.time() - t0

            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None

            metrics = compute_metrics(y_test, y_pred, y_proba)
            metrics["train_time_sec"] = round(train_time, 2)
            if train_time > 0:
                model.save(path)
            results[name] = {"metrics": metrics, "path": path}

            plot_confusion_matrix(
                y_test, y_pred,
                model_name=f"{name}_{feature_set}",
                out_dir=cm_dir,
            )
            _log.info(f"[{name}] metrics={metrics}")
        except Exception as e:
            _log.exception(f"Model {name} failed: {e}")
            results[name] = {"error": str(e)}
    return results


def load_model(name: str, feature_set: str, cfg: dict, input_dim: int | None = None, num_classes: int | None = None):
    """Reload a previously-saved model by name + feature_set."""
    models_dir = cfg["paths"]["models"]
    is_deep = name in DEEP_MODELS
    path = _model_path(models_dir, name, feature_set, is_deep)
    if is_deep:
        Cls = DEEP_MODELS[name]
        return Cls.load(path)
    else:
        Cls = CLASSICAL_MODELS[name]
        return Cls.load(path)
