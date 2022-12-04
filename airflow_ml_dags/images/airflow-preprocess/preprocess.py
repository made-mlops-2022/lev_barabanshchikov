import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--save_transformer_dir")
def preprocess(input_dir: str, output_dir: str, save_transformer_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_transformer_dir, exist_ok=True)

    scaler = StandardScaler()

    features_df = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target_df = pd.read_csv(os.path.join(input_dir, "target.csv"))

    features_df = pd.DataFrame(scaler.fit_transform(features_df))
    train_df = pd.concat([features_df, target_df], axis=1)
    train_df.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    with open(os.path.join(save_transformer_dir, "scaler.pkl"), "wb") as dump_file:
        pickle.dump(scaler, dump_file)


if __name__ == '__main__':
    preprocess()
