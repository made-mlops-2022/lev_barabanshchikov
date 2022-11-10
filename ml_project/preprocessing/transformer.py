from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), list(numeric_features)),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(categorical_features),
                ),
            ]
        )

    def fit(self, df: pd.DataFrame, target=None):
        self.transformer.fit(df, target)
        return self

    def transform(self, df: pd.DataFrame, target=None):
        df = self.transformer.transform(df)
        return df
