import os

import click as click
import gdown
import pandas as pd
import requests

ENDPOINT = "predict"
LOCALHOST = "127.0.0.1"
PORT = 8000
DS_FILENAME = "heart_cleveland_upload.csv"


def download_dataset_from_gdrive():
    url = os.getenv("DS_URL")
    if url is None:
        raise ValueError("Url environmental variable is not exported.\n"
                         "Try to execute the following in the command line:\n  export DS_URL='Given URL'")
    gdown.download_folder(url=url, quiet=True, output="data")


@click.command()
@click.option("--ds_filename", "-d", default=DS_FILENAME, type=str)
@click.option("--ip", "-i", default=LOCALHOST, type=str)
@click.option("--port", "-p", default=PORT, type=int)
def make_request(ds_filename: str, ip: str, port: int):
    if ds_filename == DS_FILENAME:
        download_dataset_from_gdrive()
    dataset = pd.read_csv(os.path.join("data", ds_filename))
    if "condition" in dataset.columns:
        dataset.drop(["condition"], axis=1, inplace=True)
    json_dict = {
        "data": dataset.values.tolist(),
        "col_names": dataset.columns.tolist()
    }
    response = requests.post(
        f"http://{ip}:{port}/{ENDPOINT}",
        json=json_dict
    )
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print(f"Prediction:\n{response.json()}")


if __name__ == "__main__":
    make_request()
