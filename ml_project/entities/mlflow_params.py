import os
from dataclasses import dataclass


@dataclass
class MlflowParams:
    run_name: str = ""
    uri: str = os.path.join(os.getcwd(), "mlruns")
    _target_: str = "ml_project.mlflow_logger.mlflow_logger.MlflowLogger"
    exp_name: str = "HeartDiseaseClassification"
