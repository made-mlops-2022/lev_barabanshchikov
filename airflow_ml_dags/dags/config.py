import os
from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "levbara",
    "email_on_failure": True,
    "email": ["levbara1@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=3)
}

DATA_VOLUME = "/home/lev/MLOps/airflow_ml_dags/data:/data"
ARTIFACTS_VOLUME = "/home/lev/MLOps/airflow_ml_dags/mlflow_logs:/data"

MLFLOW_PARAMS = {"MLFLOW_URL": os.getenv("MLFLOW_URL", "http://localhost:5001")}
