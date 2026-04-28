from sklearn.ensemble import RandomForestClassifier

from ._base import ClassicalBase


class RandomForestIDS(ClassicalBase):
    name = "random_forest"

    def _build(self, **params):
        params.setdefault("n_jobs", -1)
        params.setdefault("max_depth", 25)
        params.setdefault("min_samples_leaf", 5)
        params.setdefault("class_weight", "balanced_subsample")
        return RandomForestClassifier(**params)
