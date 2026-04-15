"""Shared base class for classical IDS models (sklearn-style wrapper)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class ClassicalBase:
    name = "base"

    def __init__(self, **params):
        self.params = params
        self.model = self._build(**params)

    def _build(self, **params):
        raise NotImplementedError

    def train(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBase":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Fallback: hard labels as one-hot-ish
        preds = self.predict(X)
        n_classes = int(preds.max()) + 1
        proba = np.zeros((len(preds), n_classes), dtype=np.float32)
        proba[np.arange(len(preds)), preds] = 1.0
        return proba

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"name": self.name, "params": self.params, "model": self.model}, f)

    @classmethod
    def load(cls, path: str) -> "ClassicalBase":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls(**obj["params"])
        inst.model = obj["model"]
        return inst
