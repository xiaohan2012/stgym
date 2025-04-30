import pytest
import torch
from torch.utils.data import DataLoader
from stgym.data_loader.brca import BRCADataset

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader import create_loader, load_dataset, STDataModule


@pytest.fixture
def mock_dl_cfg():
    return DataLoaderConfig()


@pytest.fixture
def mock_task_cfg():
    return TaskConfig(dataset_name="brca-test", type="graph-classification")


def test_load_dataset(mock_task_cfg, mock_dl_cfg):
    dataset = load_dataset(mock_task_cfg, mock_dl_cfg)
    assert isinstance(dataset, BRCADataset)


def test_create_loader(mock_task_cfg, mock_dl_cfg):
    dataset = load_dataset(mock_task_cfg, mock_dl_cfg)
    loaders = create_loader(dataset, mock_dl_cfg)
    assert len(loaders) == 3
    for loader in loaders:
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert isinstance(batch.adj_t, torch.Tensor)
        assert batch.adj_t.layout == torch.sparse_coo


def test_tl_module_init(mock_task_cfg, mock_dl_cfg):
    mod = STDataModule(mock_task_cfg, mock_dl_cfg)
    assert isinstance(mod.train_dataloader(), DataLoader)
    assert isinstance(mod.val_dataloader(), DataLoader)
    assert isinstance(mod.test_dataloader(), DataLoader)

    assert mod.num_features == 32
