import sys
import logging

sys.path.append(".")

import hydra

from ml_project.entities import Config, store_configs
from ml_project.models.predict_model import (
    get_checkpoint_path,
    load_model,
    get_data,
    make_prediction,
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")
store_configs()


def prediction_pipeline(cfg: Config):
    checkpoint_path = get_checkpoint_path(cfg)

    model = load_model(checkpoint_path)

    data = get_data(cfg.dataset)

    make_prediction(model, data, cfg.prediction.prediction_name)

    logging.info(f"Prediction saved to {cfg.prediction.prediction_name}")


@hydra.main(config_path="configs", config_name="config")
def run(cfg: Config):
    prediction_pipeline(cfg)


if __name__ == "__main__":
    run()
