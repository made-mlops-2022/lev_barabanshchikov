import os
import pickle
from datetime import datetime
from pathlib import Path

import click
import mlflow
import pandas as pd


@click.command("predict")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--transformers_dir")
def predict(input_dir: str, output_dir: str, transformers_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    latest_transformer_dir = sorted(
        list(Path(transformers_dir).iterdir()),
        key=lambda p: datetime.strptime(p.name, "%Y-%m-%d")
    ).pop()

    data = pd.read_csv(str(os.path.join(input_dir, "data.csv")))
    with open(latest_transformer_dir / "scaler.pkl", "rb") as transformer_dump:
        scaler = pickle.load(transformer_dump)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "mlflow:5001"))
    model = mlflow.pyfunc.load_model(model_uri=f"models:/LogisticRegression/Production")
    data["target"] = model.predict(scaler.transform(data))

    data.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()
