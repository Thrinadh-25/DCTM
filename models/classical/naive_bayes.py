"""Gaussian Naive Bayes baseline.

NB is intentionally kept simple as the weakest reference baseline. Network
features are heavily skewed and not independent, so this model is expected
to under-perform on multi-class — that's the academic point of including it.
"""
from sklearn.naive_bayes import GaussianNB

from ._base import ClassicalBase


class NaiveBayesIDS(ClassicalBase):
    name = "naive_bayes"

    def _build(self, **params):
        params.setdefault("var_smoothing", 1e-8)
        return GaussianNB(**params)
