import os.path
from typing import Optional, Tuple

import pandas as pd
import logging

from ml_project.entities.ds_params import DatasetParams


class Dataset:
    def __init__(self, cfg: DatasetParams) -> None:
        self.data_dir = cfg.data_dir
        self.data_path = cfg.data_path
        self.target_col = cfg.target

    def load_dataset(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if not os.path.exists(self.data_path):
            logging.info(f"Dataset doesn't exist in {self.data_path}")
            raise FileNotFoundError(f"{self.data_path}")
        df = pd.read_csv(self.data_path)
        data = df.drop(self.target_col, axis=1)
        target = df[self.target_col] if self.target_col in df.columns else None
        return data, target
