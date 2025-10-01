from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.cellcontrast_breast import CellcontrastBreastDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns for CellContrast breast dataset
    data = {
        "cellid": ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        "UMAP_1": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "UMAP_2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "donor_id": ["P10", "P10", "P10", "P11", "P11", "P11"],
        "sample_uuid": [
            "sample1",
            "sample1",
            "sample1",
            "sample2",
            "sample2",
            "sample2",
        ],
        "cell_type": [
            "B cell",
            "fibroblast",
            "B cell",
            "B cell",
            "fibroblast",
            "endothelial cell",
        ],
        "self_reported_ethnicity": [
            "European",
            "European",
            "European",
            "African American",
            "African American",
            "African American",
        ],
        "development_stage": [
            "prime adult stage",
            "prime adult stage",
            "prime adult stage",
            "prime adult stage",
            "prime adult stage",
            "prime adult stage",
        ],
        "ENSG00000223764": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "ENSG00000187608": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
        "ENSG00000260179": [0.7, 0.8, 0.9, 0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


def test_cellcontrast_breast_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/cellcontrast-breast")
        rm_dir_if_exists(data_root / "processed")
        ds = CellcontrastBreastDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples (sample1, sample2)

        # Test all samples
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (3, 3)  # 3 cells, 3 gene features
            assert data.y.shape == (3,)  # 3 labels
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (UMAP_1, UMAP_2)

        rm_dir_if_exists(data_root / "processed")


def test_cellcontrast_breast_dataset_labels(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/cellcontrast-breast")
        rm_dir_if_exists(data_root / "processed")
        ds = CellcontrastBreastDataset(root=data_root)

        # Test that labels are properly encoded as categorical codes
        all_labels = []
        for i in range(len(ds)):
            data = ds[i]
            all_labels.extend(data.y.tolist())

        # Should have encoded categorical labels (0, 1, 2 for B cell, endothelial cell, fibroblast)
        unique_labels = set(all_labels)
        assert len(unique_labels) == 3  # 3 unique cell types in mock data
        assert all(isinstance(label, int) for label in unique_labels)

        rm_dir_if_exists(data_root / "processed")
