import os
from datetime import timedelta

from docker.types import Mount

DEFAULT_ARGS = {
    "owner": "levbara",
    "email_on_failure": True,
    "email": ["levbara1@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=3)
}

DATA_MOUNT = Mount(source="/home/lev/MLOps/airflow_ml_dags/data/", target="/data", type="bind")
ARTIFACTS_MOUNT = Mount(source="/home/lev/MLOps/airflow_ml_dags/mlflow_logs/", target="/mlruns", type="bind")

MLFLOW_PARAMS = {"MLFLOW_URL": os.getenv("MLFLOW_URL", "http://localhost:5001")}
