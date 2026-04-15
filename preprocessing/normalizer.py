"""MinMax normalizer fit on train only; persists to disk."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.logger import get_logger

_log = get_logger("normalizer")


class Normalizer:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.columns: Optional[list[str]] = None

    def fit_transform(self, X, columns: Optional[list[str]] = None):
        self.columns = columns
        arr = self.scaler.fit_transform(np.asarray(X, dtype=np.float32))
        return arr.astype(np.float32)

    def transform(self, X):
        arr = self.scaler.transform(np.asarray(X, dtype=np.float32))
        return arr.astype(np.float32)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "columns": self.columns}, f)
        _log.info(f"Scaler saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls()
        inst.scaler = obj["scaler"]
        inst.columns = obj.get("columns")
        return inst
