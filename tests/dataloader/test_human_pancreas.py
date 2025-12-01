from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.human_pancreas import HumanPancreasDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples from different developmental stages
    data = {
        "barcode": ["CELL1-1", "CELL2-1", "CELL3-1", "CELL4-1", "CELL5-1", "CELL6-1"],
        "in_tissue": [1, 1, 1, 1, 1, 1],
        "array_row": [10, 11, 12, 20, 21, 22],
        "array_col": [10, 11, 12, 20, 21, 22],
        "pxl_row_in_fullres": [1000, 1100, 1200, 2000, 2100, 2200],
        "pxl_col_in_fullres": [1000, 1100, 1200, 2000, 2100, 2200],
        "stage": ["12PCW", "12PCW", "12PCW", "15PCW", "15PCW", "15PCW"],
        "section": ["1", "1", "1", "1", "1", "1"],
        "sample_id": [
            "12PCW_S1",
            "12PCW_S1",
            "12PCW_S1",
            "15PCW_S1",
            "15PCW_S1",
            "15PCW_S1",
        ],
        "cell_type": ["acinar", "ductal", "immune", "acinar", "beta", "alpha"],
        "x_coord": [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 2200.0],
        "y_coord": [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 2200.0],
        "gene_SPINK1": [10.5, 20.3, 5.1, 15.2, 8.7, 12.4],
        "gene_CLPS": [8.2, 15.6, 3.8, 12.1, 6.5, 9.3],
        "gene_CEL": [12.1, 18.9, 7.2, 14.5, 10.8, 11.6],
        "prop_acinar": [0.8, 0.1, 0.0, 0.7, 0.0, 0.0],
        "prop_ductal": [0.1, 0.8, 0.0, 0.2, 0.0, 0.0],
        "prop_immune": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        "prop_beta": [0.0, 0.0, 0.0, 0.0, 0.8, 0.1],
        "prop_alpha": [0.0, 0.0, 0.0, 0.0, 0.1, 0.8],
    }
    return pd.DataFrame(data)


def test_human_pancreas_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/human-pancreas")
        rm_dir_if_exists(data_root / "processed")
        ds = HumanPancreasDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples (12PCW_S1, 15PCW_S1)

        # Check label encoding
        assert ds.y.min() == 0
        assert (
            ds.y.max() == 4
        )  # 5 unique cell types (acinar=0, alpha=1, beta=2, ductal=3, immune=4) -> 0,1,2,3,4

        for i, data in enumerate(ds):
            assert data.x.shape[0] == 3  # 3 cells per sample
            assert data.x.shape[1] == 8  # 3 gene features + 5 prop features
            assert data.y.shape == (3,)  # 3 labels
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)

            # Check that positions are properly extracted
            assert data.pos.min() >= 1000.0  # coordinates should be in pixel space

            # Check that features are numeric
            assert data.x.dtype.is_floating_point

        # Check that samples have different characteristics
        sample1_labels = set(ds[0].y.numpy())
        sample2_labels = set(ds[1].y.numpy())
        assert (
            sample1_labels != sample2_labels
        )  # Different samples should have different label distributions

        rm_dir_if_exists(data_root / "processed")


def test_human_pancreas_dataset_cell_types():
    # Test with actual cell type names to ensure proper mapping
    cell_types = ["acinar", "ductal", "immune", "beta", "alpha"]
    mock_data = {
        "sample_id": ["sample1"] * 5,
        "cell_type": cell_types,
        "x_coord": [100.0, 200.0, 300.0, 400.0, 500.0],
        "y_coord": [100.0, 200.0, 300.0, 400.0, 500.0],
        "gene_TEST1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "gene_TEST2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "prop_acinar": [1.0, 0.0, 0.0, 0.0, 0.0],
        "prop_ductal": [0.0, 1.0, 0.0, 0.0, 0.0],
        "prop_immune": [0.0, 0.0, 1.0, 0.0, 0.0],
        "prop_beta": [0.0, 0.0, 0.0, 1.0, 0.0],
        "prop_alpha": [0.0, 0.0, 0.0, 0.0, 1.0],
        # Add required columns
        "barcode": ["CELL1", "CELL2", "CELL3", "CELL4", "CELL5"],
        "in_tissue": [1, 1, 1, 1, 1],
        "array_row": [1, 2, 3, 4, 5],
        "array_col": [1, 2, 3, 4, 5],
        "pxl_row_in_fullres": [100, 200, 300, 400, 500],
        "pxl_col_in_fullres": [100, 200, 300, 400, 500],
        "stage": ["12PCW"] * 5,
        "section": ["1"] * 5,
    }
    mock_df = pd.DataFrame(mock_data)

    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/human-pancreas-types")
        rm_dir_if_exists(data_root / "processed")
        ds = HumanPancreasDataset(root=data_root)

        assert len(ds) == 1  # one sample
        data = ds[0]

        # Check that we have 5 different cell types (0, 1, 2, 3, 4)
        unique_labels = sorted(data.y.numpy())
        assert unique_labels == [0, 1, 2, 3, 4]

        # Check feature dimensions
        assert data.x.shape == (5, 7)  # 5 cells, 2 gene + 5 prop features

        rm_dir_if_exists(data_root / "processed")
