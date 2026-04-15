from sklearn.naive_bayes import GaussianNB

from ._base import ClassicalBase


class NaiveBayesIDS(ClassicalBase):
    name = "naive_bayes"

    def _build(self, **params):
        return GaussianNB(**params)
