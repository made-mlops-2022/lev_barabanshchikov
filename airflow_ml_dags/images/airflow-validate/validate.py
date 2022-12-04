import json
import os

import click
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.command("validate")
@click.option("--data_dir")
@click.option("--metrics_dir")
def validate(data_dir: str, metrics_dir: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "mlflow:5001"))
    mlflow.set_experiment(f"Validate_LogReg")
    client = MlflowClient()

    os.makedirs(metrics_dir, exist_ok=True)

    last_model_params = sorted(
        client.search_model_versions("name='LogisticRegression'"),
        key=lambda x: x.last_updated_timestamp
    )[-1]
    model = mlflow.sklearn.load_model(f"models:/{last_model_params.name}/{last_model_params.version}")

    valid_data = pd.read_csv(os.path.join(data_dir, "validate.csv"))
    valid_target = valid_data.target
    valid_features = valid_data.drop("target", axis=1)

    pred = model.predict(valid_features)
    metrics = {
        "accuracy": accuracy_score(valid_target, pred),
        "f1_score": f1_score(valid_target, pred),
        "AUC": roc_auc_score(valid_target, pred),
    }
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as dump_file:
        json.dump(metrics, dump_file)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)

    client.transition_model_version_stage(
        name=last_model_params.name,
        version=last_model_params.version,
        stage="Production"
    )


if __name__ == "__main__":
    validate()
