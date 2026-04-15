from .decision_tree import DecisionTreeIDS
from .naive_bayes import NaiveBayesIDS
from .logistic_regression import LogisticRegressionIDS
from .random_forest import RandomForestIDS
from .xgboost_model import XGBoostIDS
from .svm_model import SVMIDS

CLASSICAL_MODELS = {
    "decision_tree": DecisionTreeIDS,
    "naive_bayes": NaiveBayesIDS,
    "logistic_regression": LogisticRegressionIDS,
    "random_forest": RandomForestIDS,
    "xgboost": XGBoostIDS,
    "svm": SVMIDS,
}
