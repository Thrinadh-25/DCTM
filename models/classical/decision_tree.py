from sklearn.tree import DecisionTreeClassifier

from ._base import ClassicalBase


class DecisionTreeIDS(ClassicalBase):
    name = "decision_tree"

    def _build(self, **params):
        params.setdefault("class_weight", "balanced")
        params.setdefault("min_samples_split", 20)
        params.setdefault("min_samples_leaf", 10)
        params.setdefault("criterion", "gini")
        return DecisionTreeClassifier(**params)
