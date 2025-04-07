import pytest
from torch.utils.data import DataLoader
from torch_geometric.datasets import BA2MotifDataset

from stgym.config_schema import DataLoaderConfig
from stgym.data_loader import create_loader, load_dataset


@pytest.fixture
def mock_cfg():
    return DataLoaderConfig(dataset_name="ba2motif")


def test_load_dataset(mock_cfg):
    dataset = load_dataset(mock_cfg)
    isinstance(dataset, BA2MotifDataset)


def test_create_loader(mock_cfg):
    loaders = create_loader(mock_cfg)
    assert len(loaders) == 3
    for loader in loaders:
        assert isinstance(loader, DataLoader)
