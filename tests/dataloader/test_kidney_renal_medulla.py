from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.kidney_renal_medulla import KidneyRenalMedullaDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        "Unnamed: 0": ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        "UMAP_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "UMAP_2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "donor_id": ["donor1", "donor1", "donor1", "donor2", "donor2", "donor2"],
        "sample_uuid": [
            "sample1",
            "sample1",
            "sample1",
            "sample2",
            "sample2",
            "sample2",
        ],
        "author_cell_type": ["type1", "type2", "type1", "type1", "type2", "type1"],
        "cell_type": [
            "full_type1",
            "full_type2",
            "full_type1",
            "full_type1",
            "full_type2",
            "full_type1",
        ],
        "sex": ["M", "M", "M", "F", "F", "F"],
        "tissue": ["kidney", "kidney", "kidney", "kidney", "kidney", "kidney"],
        "development_stage": ["adult", "adult", "adult", "adult", "adult", "adult"],
        "ENSG00000268895": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "ENSG00000115977": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "ENSG00000125257": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    }
    return pd.DataFrame(data)


def test_kidney_renal_medulla_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/kidney-renal-medulla")
        rm_dir_if_exists(data_root / "processed")
        ds = KidneyRenalMedullaDataset(root=data_root)
        assert (
            len(ds) == 2
        )  # there are two graph samples (2 different sample_uuid values)

        # Test all samples
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (3, 3)  # 3 cells per sample, 3 gene features
            assert data.y.shape == (3,)  # 3 labels per sample
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (UMAP_1, UMAP_2)
            # Check that labels are categorical codes (0, 1, etc.)
            assert data.y.min() >= 0
            assert data.y.max() < 2  # 2 unique cell types in mock data

        rm_dir_if_exists(data_root / "processed")
