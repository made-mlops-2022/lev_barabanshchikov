import sys
import logging

sys.path.append(".")

import hydra

from ml_project.entities import Config, store_configs
from ml_project.models.train_model import (
    get_data,
    save_artifacts,
    train_model,
    get_score,
    initialize_logger
)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s", datefmt="%d:%m:%Y|%H:%M:%S")
store_configs()


def train_pipeline(cfg: Config) -> None:
    logger = initialize_logger(cfg)

    train_data, test_data, train_target, test_target = get_data(cfg)

    model = train_model(cfg, train_data, train_target)

    score = get_score(cfg, model, test_data, test_target)

    save_artifacts(cfg, model, score, logger)


@hydra.main(config_path="configs", config_name="config")
def run(cfg: Config):
    train_pipeline(cfg)


if __name__ == "__main__":
    run()
