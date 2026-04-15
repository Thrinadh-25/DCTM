from sklearn.tree import DecisionTreeClassifier

from ._base import ClassicalBase


class DecisionTreeIDS(ClassicalBase):
    name = "decision_tree"

    def _build(self, **params):
        return DecisionTreeClassifier(**params)
