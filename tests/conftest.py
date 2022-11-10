import pytest
from hydra import initialize, compose

from .gen_fake_dataset import gen_dataset


@pytest.fixture
def config():
    with initialize(config_path="../ml_project/configs"):
        cfg = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=["common.project_dir=${hydra.runtime.cwd}"],
        )
        if "mlflow" in cfg:
            cfg.mlflow = None
    return cfg


@pytest.fixture
def fake_dataset():
    df = gen_dataset(10)
    return df


@pytest.fixture
def fake_dataset_path(tmpdir, fake_dataset):
    dataset_fio = tmpdir.join("fake_dataset.csv")
    fake_dataset.to_csv(dataset_fio, index=False)
    return str(dataset_fio)
