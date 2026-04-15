"""SVM wrapper that automatically subsamples large training sets.

sklearn's SVC is O(n^2)–O(n^3) so on CICIDS (millions of rows) we must
subsample. Sample size is configurable via config['classical']['svm']['max_train_samples'].
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from utils.logger import get_logger

from ._base import ClassicalBase

_log = get_logger("svm")


class SVMIDS(ClassicalBase):
    name = "svm"

    def _build(self, **params):
        self.max_train_samples = int(params.pop("max_train_samples", 50000))
        return SVC(**params)

    def train(self, X, y):
        n = len(X)
        if n > self.max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]
            _log.info(f"SVM subsample: {n} -> {self.max_train_samples}")
        self.model.fit(X, y)
        return self
