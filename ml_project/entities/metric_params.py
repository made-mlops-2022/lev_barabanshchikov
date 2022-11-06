from dataclasses import dataclass


@dataclass
class AccuracyParams:
    _target_: str = "sklearn.metrics.accuracy_score"


@dataclass
class F1Params:
    _target_: str = "sklearn.metrics.f1_score"


@dataclass
class RocAucParams:
    _target_: str = "sklearn.metrics.roc_auc_score"
