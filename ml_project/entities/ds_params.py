import os
from dataclasses import dataclass


@dataclass
class DatasetParams:
    data_dir: str = os.path.join(os.getcwd(), "data", "raw")
    data_path: str = os.path.join(os.getcwd(), "data", "raw", "heart_cleveland_upload.csv")
    val_size: float = 0.2
    random_state: int = 42
    target: str = "condition"
