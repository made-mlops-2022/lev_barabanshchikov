import os
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

from ml_project.entities import (
    DatasetParams,
    FeatureParams,
    AccuracyParams,
    F1Params,
    RocAucParams,
    MlflowParams,
    LogisticRegressionParams,
    RandomForestParams,
    PredictParams
)


@dataclass
class CommonParams:
    run_name: str = ""
    project_dir: str = os.getcwd()
    artifacts_dir: str = "outputs"
    cp_path: str = "weights"


@dataclass
class Config:
    common: CommonParams = CommonParams()
    dataset: DatasetParams = DatasetParams()
    features: Any = FeatureParams()
    mlflow: Any = MlflowParams()
    metric: Any = F1Params()
    model: Any = RandomForestParams()
    prediction: Any = PredictParams()


def store_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="configs/config", node=Config)
    cs.store(group="configs/common", name="common", node=CommonParams)
    cs.store(group="configs/dataset", name="ds", node=DatasetParams)
    cs.store(group="configs/features", name="features", node=FeatureParams)
    cs.store(group="configs/metric", name="accuracy", node=AccuracyParams)
    cs.store(group="configs/metric", name="f1", node=F1Params)
    cs.store(group="configs/metric", name="roc_auc", node=RocAucParams)
    cs.store(group="configs/mlflow", name="mlflow", node=MlflowParams)
    cs.store(group="configs/model", name="logistic_regression", node=LogisticRegressionParams)
    cs.store(group="configs/model", name="random_forest", node=RandomForestParams)
    cs.store(group="configs/prediction", name="predict", node=PredictParams)
