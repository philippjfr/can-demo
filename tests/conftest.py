import pytest

from candetect.data import DataConfig, generate_dataset, make_splits


@pytest.fixture(scope="session")
def small_dataset():
    return generate_dataset(DataConfig(n_images=120, seed=3))


@pytest.fixture(scope="session")
def small_splits(small_dataset):
    return make_splits(small_dataset.labels, seed=3)
