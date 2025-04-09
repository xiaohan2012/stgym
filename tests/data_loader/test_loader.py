import pytest
from torch.utils.data import DataLoader
from torch_geometric.datasets import BA2MotifDataset

from stgym.config_schema import DataLoaderConfig
from stgym.data_loader import create_loader, load_dataset, STDataModule


@pytest.fixture
def mock_cfg():
    return DataLoaderConfig(dataset_name="ba2motif")


def test_load_dataset(mock_cfg):
    dataset = load_dataset(mock_cfg)
    isinstance(dataset, BA2MotifDataset)


def test_create_loader(mock_cfg):
    dataset = load_dataset(mock_cfg)
    loaders = create_loader(dataset, mock_cfg)
    assert len(loaders) == 3
    for loader in loaders:
        assert isinstance(loader, DataLoader)



def test_tl_module_init(mock_cfg):
    mod = STDataModule(mock_cfg)
    assert isinstance(mod.train_dataloader(), DataLoader)
    assert isinstance(mod.val_dataloader(), DataLoader)
    assert isinstance(mod.test_dataloader(), DataLoader)

    assert mod.num_features == 10
