from sklearn.ensemble import RandomForestClassifier

from ._base import ClassicalBase


class RandomForestIDS(ClassicalBase):
    name = "random_forest"

    def _build(self, **params):
        return RandomForestClassifier(**params)
