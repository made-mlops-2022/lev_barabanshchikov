from dataclasses import dataclass


@dataclass
class LogisticRegressionParams:
    _target_: str = "sklearn.linear_model.LogisticRegression"
    penalty: str = "l2"
    solver: str = "liblinear"
    C: float = 1.0
    max_iter: int = 800


@dataclass
class RandomForestParams:
    _target_: str = "sklearn.ensemble.RandomForestClassifier"
    n_estimators: int = 500
    max_depth: int = 5
