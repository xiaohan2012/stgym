import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader import (
    STDataModule,
    STKfoldDataModule,
    create_kfold_loader,
    create_loader,
    load_dataset,
)
from stgym.data_loader.brca import BRCADataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_dl_cfg():
    return DataLoaderConfig()


@pytest.fixture
def mock_task_cfg():
    return TaskConfig(
        dataset_name="brca-test", type="graph-classification", num_classes=2
    )


def test_load_dataset(mock_task_cfg, mock_dl_cfg):
    dataset = load_dataset(mock_task_cfg, mock_dl_cfg)
    assert isinstance(dataset, BRCADataset)


def test_create_loader(mock_task_cfg, mock_dl_cfg):
    dataset = load_dataset(mock_task_cfg, mock_dl_cfg)
    loaders = create_loader(dataset, mock_dl_cfg)
    assert len(dataset) > 0

    assert len(loaders) == 3
    for loader in loaders:
        assert isinstance(loader, DataLoader)
        if len(loader) > 0:
            batch = next(iter(loader))
            assert isinstance(batch.adj_t, torch.Tensor)
            assert batch.adj_t.layout == torch.sparse_coo

    # ensure 0 mean and 1 std
    train_loader = loaders[0]
    train_x = np.concatenate([data.x.numpy() for data in train_loader])
    np.testing.assert_allclose(train_x.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(train_x.std(axis=0), 1, atol=1e-5)


def test_create_kfold_loader(mock_task_cfg, mock_dl_cfg):
    # Use 2 folds for simple validation
    mock_dl_cfg_0, mock_dl_cfg_1 = mock_dl_cfg.model_copy(
        deep=True
    ), mock_dl_cfg.model_copy(deep=True)
    mock_dl_cfg_0.split = DataLoaderConfig.KFoldSplitConfig(num_folds=2, split_index=0)
    mock_dl_cfg_1.split = DataLoaderConfig.KFoldSplitConfig(num_folds=2, split_index=1)
    dataset = load_dataset(mock_task_cfg, mock_dl_cfg)

    # Get loaders for both folds
    loaders_at_fold_0 = create_kfold_loader(dataset, mock_dl_cfg_0)
    loaders_at_fold_1 = create_kfold_loader(dataset, mock_dl_cfg_1)

    # Extract datasets for easier comparison
    train_0, val_0, test_0 = loaders_at_fold_0
    train_1, val_1, test_1 = loaders_at_fold_1

    # Test 1: Validation sets should be disjoint
    val_0_indices = set(val_0.dataset.indices)
    val_1_indices = set(val_1.dataset.indices)
    assert len(val_0_indices.intersection(val_1_indices)) == 0

    # Test 2: Together, validation sets should cover entire dataset
    assert len(val_0_indices.union(val_1_indices)) == len(dataset)

    # Test 3: Each fold's train set should be the other fold's validation set
    train_0_indices = set(train_0.dataset.indices)
    train_1_indices = set(train_1.dataset.indices)
    assert train_0_indices == val_1_indices  # Fold 0 train = Fold 1 val
    assert train_1_indices == val_0_indices  # Fold 1 train = Fold 0 val

    # Test 4: Test dataset should be same as validation (k-fold CV pattern)
    test_0_indices = set(test_0.dataset.indices)
    test_1_indices = set(test_1.dataset.indices)
    assert test_0_indices == val_0_indices
    assert test_1_indices == val_1_indices

    # Test 5: Verify correct sizes for 2-fold split
    expected_fold_size = len(dataset) // 2
    assert len(val_0_indices) == pytest.approx(expected_fold_size, abs=1)
    assert len(val_1_indices) == pytest.approx(
        len(dataset) - expected_fold_size, abs=1
    )  # Handle remainder

    # Test 6: Verify DataLoader properties are preserved
    assert train_0.batch_size == mock_dl_cfg.batch_size
    assert val_0.batch_size == mock_dl_cfg.batch_size
    assert test_0.batch_size == mock_dl_cfg.batch_size

    # Test 7: Verify data integrity - should be able to iterate
    for data_set in [train_0, val_0, test_0, train_1, val_1, test_1]:
        batch = next(iter(data_set))
        assert hasattr(batch, "x")  # Should have node features
        assert hasattr(batch, "y")  # Should have labels
        assert hasattr(batch, "adj_t")  # adj matrix

    # features in training data have 0 mean and unit std
    train_x = np.concatenate([data.x.numpy() for data in train_0])
    np.testing.assert_allclose(train_x.mean(axis=0), 0)
    np.testing.assert_allclose(train_x.std(axis=0), 1)

    # train_features = train_0.dataset.dataset.x[train_0.dataset.indices]
    # np.testing.assert_allclose(train_features.mean(axis=0).numpy(), 0,
    #                            atol=1e-6)
    # np.testing.assert_allclose(train_features.std(axis=0).numpy(), 1,
    #                            atol=1e-6)


def test_tl_module_init(mock_task_cfg, mock_dl_cfg):
    mod = STDataModule(mock_task_cfg, mock_dl_cfg)
    assert isinstance(mod.train_dataloader(), DataLoader)
    assert isinstance(mod.val_dataloader(), DataLoader)
    assert isinstance(mod.test_dataloader(), DataLoader)


class TestSTKfoldDataModule:
    num_folds = 3

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_basic(self, mock_task_cfg, mock_dl_cfg, k):
        mock_dl_cfg.split = DataLoaderConfig.KFoldSplitConfig(
            num_folds=self.num_folds, split_index=k
        )
        mod = STKfoldDataModule(mock_task_cfg, mock_dl_cfg)
        assert isinstance(mod.train_dataloader(), DataLoader)
        assert isinstance(mod.val_dataloader(), DataLoader)
        assert isinstance(mod.test_dataloader(), DataLoader)


def teardown_module(module):
    """Teardown function called after all tests in this module."""
    rm_dir_if_exists("tests/data/brca-test/processed")
