"""Multinomial Logistic Regression with `saga` solver for scalability.

`saga` supports multiclass natively, parallelizes via n_jobs, and converges
much faster than `lbfgs` on large balanced datasets. We also subsample the
training set if it is excessively large, so a single classifier doesn't
dominate the wall-clock budget.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.logger import get_logger

from ._base import ClassicalBase

_log = get_logger("logreg")


class LogisticRegressionIDS(ClassicalBase):
    name = "logistic_regression"

    def _build(self, **params):
        params = dict(params)
        self.max_train_samples = int(params.pop("max_train_samples", 500_000))
        params.setdefault("solver", "saga")
        params.setdefault("max_iter", 300)
        params.setdefault("n_jobs", -1)
        params.setdefault("class_weight", "balanced")
        params.setdefault("C", 1.0)
        return LogisticRegression(**params)

    def train(self, X, y):
        n = len(X)
        if n > self.max_train_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]
            _log.info(f"LogReg subsample: {n} -> {self.max_train_samples}")
        self.model.fit(X, y)
        return self
