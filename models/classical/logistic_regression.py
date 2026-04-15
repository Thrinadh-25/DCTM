from sklearn.linear_model import LogisticRegression

from ._base import ClassicalBase


class LogisticRegressionIDS(ClassicalBase):
    name = "logistic_regression"

    def _build(self, **params):
        return LogisticRegression(**params)
