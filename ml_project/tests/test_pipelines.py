import os
import pytest
from ml_project.train_pipeline import train_pipeline
from ml_project.predict import prediction_pipeline


def test_train_pipeline(tmp_path, config, fake_dataset_path):
    checkpoint_path = str(tmp_path.joinpath(config.common.cp_path))
    model_path = os.path.join(checkpoint_path, "model.pkl")

    config.dataset.data_path = fake_dataset_path
    config.common.cp_path = checkpoint_path
    train_pipeline(config)

    assert os.path.exists(model_path), "Model isn't saved!"


@pytest.fixture
def artifacts_path(tmp_path, config, fake_dataset_path):
    checkpoint_path = str(tmp_path.joinpath(config.common.cp_path))
    config.dataset.data_path = fake_dataset_path
    config.common.cp_path = checkpoint_path
    train_pipeline(config)
    return checkpoint_path


def test_prediction_pipeline(
    tmp_path, artifacts_path, config, fake_dataset, fake_dataset_path
):
    prediction_path = str(tmp_path.joinpath(config.prediction.prediction_name))
    config.dataset.data_path = fake_dataset_path
    config.common.cp_path = artifacts_path
    config.prediction.prediction_name = prediction_path

    prediction_pipeline(config)

    assert os.path.exists(prediction_path), "There is no prediction file!"

    with open(prediction_path, "r") as fin:
        lines = fin.readlines()

    assert len(lines) == len(
        fake_dataset
    ), f"Prediction size ({len(lines)}) is not equal with dataset size ({len(fake_dataset)})"
