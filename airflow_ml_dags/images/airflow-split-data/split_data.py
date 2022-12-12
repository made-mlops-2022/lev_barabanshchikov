import os.path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input_dir")
@click.option("--output_dir")
def split_data(input_dir: str, output_dir: str):
    dataframe = pd.read_csv(os.path.join(input_dir, "data.csv"))

    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42, shuffle=True)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"))
    test_df.to_csv(os.path.join(output_dir, "validate.csv"))


if __name__ == "__main__":
    split_data()
