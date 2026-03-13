from unittest.mock import patch

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


@patch("stgym.data_loader.load_dataset")
class TestSTDataModuleNaNDetection:
    """Test NaN detection functionality in STDataModule"""

    @property
    def mock_dataset_with_nans(self):
        """Create a mock dataset with NaN values in features"""
        from torch_geometric.data import Data

        # Create a simple mock dataset with NaN values
        class MockDataset:
            def __init__(self):
                # Create sample data with NaN values
                x = torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    dtype=torch.float,
                )
                x[0, 0] = float("nan")  # Add NaN to first feature of first sample
                x[1, 2] = float("nan")  # Add NaN to third feature of second sample

                self.data = Data(
                    x=x,
                    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                    y=torch.tensor([0, 1, 0]),
                )

            def to(self, device):
                self.data = self.data.to(device)
                return self

            @property
            def x(self):
                return self.data.x

            def __len__(self):
                return 3

        return MockDataset()

    def test_nan_detection_raises_error(
        self, mock_load_dataset, mock_task_cfg, mock_dl_cfg
    ):
        """Test that STDataModule raises ValueError when dataset contains NaN values"""
        # Setup mock to return dataset with NaN values
        mock_load_dataset.return_value = self.mock_dataset_with_nans

        # Test STDataModule raises error on NaN detection
        with pytest.raises(
            ValueError, match=r"Dataset has \d+/\d+ \(\d+\.\d+%\) NaN feature values"
        ):
            STDataModule(mock_task_cfg, mock_dl_cfg)

        # Test STKfoldDataModule also raises error on NaN detection
        mock_dl_cfg.split = DataLoaderConfig.KFoldSplitConfig(
            num_folds=2, split_index=0
        )
        with pytest.raises(
            ValueError, match=r"Dataset has \d+/\d+ \(\d+\.\d+%\) NaN feature values"
        ):
            STKfoldDataModule(mock_task_cfg, mock_dl_cfg)


class TestDropLastBehavior:
    """Test for issue #42: drop_last=False for validation/test sets to ensure val_loss availability."""

    @property
    def small_mock_dataset(self):
        """Create a small mock dataset to reproduce drop_last issues."""
        from torch_geometric.data import Data

        class MockDataset:
            def __init__(self, num_samples):
                self.data_list = []
                for i in range(num_samples):
                    # Create simple mock data points
                    x = torch.randn(10, 5)  # 10 nodes, 5 features
                    y = torch.randint(0, 21, (10,))  # 21 classes (like human-intestine)
                    pos = torch.randn(10, 2)  # 2D positions
                    self.data_list.append(Data(x=x, y=y, pos=pos))

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                return self.data_list[idx]

        return MockDataset

    @pytest.mark.parametrize(
        "num_samples, batch_size, num_folds, expected_val_samples",
        [
            (8, 32, 8, 1),  # Original issue #42 scenario
            (6, 20, 3, 2),  # 3-fold with 6 samples
            (10, 15, 5, 2),  # 5-fold with 10 samples
        ],
    )
    def test_kfold_large_batch_size(
        self, num_samples, batch_size, num_folds, expected_val_samples
    ):
        """
        Test k-fold CV with large batch size doesn't drop validation data.

        Before the fix, validation sets smaller than batch_size would be completely
        dropped, causing early stopping to fail due to missing val_loss.
        """
        dataset = self.small_mock_dataset(num_samples)

        config = DataLoaderConfig(
            batch_size=batch_size,
            split=DataLoaderConfig.KFoldSplitConfig(num_folds=num_folds, split_index=0),
            device="cpu",
            graph_const="knn",
            knn_k=5,
        )

        train_loader, val_loader, test_loader = create_kfold_loader(dataset, config)

        # Validation and test loaders must have data (this was the bug)
        assert len(val_loader) > 0, "Validation loader should not be empty"
        assert len(test_loader) > 0, "Test loader should not be empty"

        # Count actual samples to verify expected behavior
        val_samples = sum(len(batch) for batch in val_loader)
        assert (
            val_samples == expected_val_samples
        ), f"Expected {expected_val_samples} validation samples, got {val_samples}"

        # Verify we can iterate through data (simulates PyTorch Lightning training)
        for batch in val_loader:
            assert batch.x is not None
            assert batch.y is not None
            assert len(batch) > 0

    def test_regular_split_large_batch_size(self):
        """Test regular train/val/test split with large batch size preserves val/test data."""
        dataset = self.small_mock_dataset(6)

        config = DataLoaderConfig(
            batch_size=10,  # Larger than any split
            split=DataLoaderConfig.DataSplitConfig(
                train_ratio=0.5,  # 3 samples
                val_ratio=0.25,  # ~1.5 → 2 samples
                test_ratio=0.25,  # ~1.5 → 1 sample
            ),
            device="cpu",
            graph_const="knn",
            knn_k=5,
        )

        train_loader, val_loader, test_loader = create_loader(dataset, config)

        # All loaders should have data (before fix, small val/test sets would be dropped)
        assert len(val_loader) > 0, "Validation loader should not be empty"
        assert len(test_loader) > 0, "Test loader should not be empty"

        # Verify data accessibility
        val_samples = sum(len(batch) for batch in val_loader)
        test_samples = sum(len(batch) for batch in test_loader)

        assert val_samples > 0, f"Expected validation samples, got {val_samples}"
        assert test_samples > 0, f"Expected test samples, got {test_samples}"


def teardown_module(module):
    """Teardown function called after all tests in this module."""
    rm_dir_if_exists("tests/data/brca-test/processed")
