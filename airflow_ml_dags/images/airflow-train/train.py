import os
import pickle

import click
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_PARAMS = {"C": 10, "solver": "sag"}


@click.command("train")
@click.option("--data_dir")
@click.option("--save_model_dir")
def train(data_dir: str, save_model_dir: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "mlflow:5001"))
    mlflow.set_experiment(f"Train_LogReg")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        os.makedirs(save_model_dir, exist_ok=True)

        model = LogisticRegression(**MODEL_PARAMS)

        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
        train_target = train_data.target
        train_features = train_data.drop("target", axis=1)

        model.fit(train_features, train_target)

        with open(os.path.join(save_model_dir, "model.pkl"), "wb") as model_dump_file:
            pickle.dump(model, model_dump_file)

        mlflow.sklearn.log_model(model, artifact_path="models", registered_model_name="LogisticRegression")


if __name__ == '__main__':
    train()
