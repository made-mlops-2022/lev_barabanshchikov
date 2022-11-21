import numpy as np
from ml_project.dataset.dataset import Dataset
from ml_project.preprocessing.transformer import Transformer


def test_data_transformer(config, fake_dataset):
    transformer = Transformer(
        config.features.numeric_features, config.features.categorical_features
    )
    data = fake_dataset.drop(config.dataset.target, axis=1)
    target = fake_dataset[config.dataset.target]

    data = transformer.fit_transform(data, target)

    print(data)
    assert np.allclose(
        data[:, : len(config.features.numeric_features)].mean(axis=0), 0, atol=1e-6
    ) and np.allclose(
        data[:, : len(config.features.numeric_features)].std(axis=0), 1, atol=1e-6
    ), "Numeric features were not standardized"


def test_dataset_loading(config, fake_dataset, fake_dataset_path):
    config.dataset.data_path = fake_dataset_path
    data, target = Dataset(config.dataset).load_dataset()

    assert len(data) == len(
        fake_dataset
    ), f"dataset length should be {len(fake_dataset)} (get: {len(data)})"
    assert np.all(
        target == fake_dataset[config.dataset.target]
    ), "targets are different from ground truth"
