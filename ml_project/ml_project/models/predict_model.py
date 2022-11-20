import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ml_project.entities import Config, DatasetParams
from ml_project.dataset.dataset import Dataset
from ml_project.utils.tech_magic import get_last_artifacts_path, load_pickle


def get_checkpoint_path(cfg: Config) -> Path:
    checkpoint_path = (
        Path(cfg.common.project_dir)
        / cfg.common.artifacts_dir
        / cfg.prediction.run_name
        / cfg.common.cp_path
    )

    if not checkpoint_path.exists():
        logging.debug(
            f"There are no artifacts in {checkpoint_path}. Trying to get last artifacts path"
        )
        last_artifacts_dir = get_last_artifacts_path(cfg)
        if last_artifacts_dir:
            checkpoint_path = last_artifacts_dir / cfg.common.cp_path
            logging.debug(f"Using last experiment with artifacts {checkpoint_path}")
        else:
            raise FileNotFoundError("There are no artifacts in stated path")
    return checkpoint_path


def load_model(checkpoint_path: Path) -> Any:
    model = load_pickle(str(checkpoint_path / "model.pkl"))
    return model


def get_data(cfg: DatasetParams) -> pd.DataFrame:
    data, _ = Dataset(cfg).load_dataset()
    return data


def make_prediction(model: Any, data: pd.DataFrame, save_path: str):
    prediction = model.predict(data)
    with open(save_path, "w") as f:
        f.write("\n".join(prediction.astype(str)))
