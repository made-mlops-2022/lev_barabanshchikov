import pickle
import pandas as pd

from validation import InputData


def extract_model():
    with open("online_inference/model/model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def extract_data(request: InputData) -> pd.DataFrame:
    return pd.DataFrame(
        data=request.data,
        columns=request.col_names
    )
