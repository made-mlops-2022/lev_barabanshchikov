import os

import click
from sklearn.datasets import load_breast_cancer


@click.command("get")
@click.argument("output_dir")
def get_data(output_dir: str):
    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    os.makedirs(output_dir, exist_ok=True)

    features.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    get_data()
