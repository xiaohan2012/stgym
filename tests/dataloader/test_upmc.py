from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.upmc import UpmcDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        "CELL_ID": [1, 2, 3, 4, 5, 6],
        "X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "SIZE": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "CELL_TYPE": ["Tumor", "B cell", "Tumor", "Tumor", "B cell", "Tumor"],
        "reg_id": [
            "UPMC_c001_v001_r001_reg001",
            "UPMC_c001_v001_r001_reg001",
            "UPMC_c001_v001_r001_reg001",
            "UPMC_c001_v001_r001_reg002",
            "UPMC_c001_v001_r001_reg002",
            "UPMC_c001_v001_r001_reg002",
        ],
        "CD11b": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "CD14": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
        "CD15": [0.7, 0.8, 0.9, 0.7, 0.8, 0.9],
        "CD163": [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
    }
    return pd.DataFrame(data)


def test_upmc_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/upmc")
        rm_dir_if_exists(data_root / "processed")
        ds = UpmcDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples
        # Test all regions
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (
                3,
                4,
            )  # 3 cells, 4 features (CD11b, CD14, CD15, CD163)
            assert data.y.shape == (3,)  # 3 labels
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (X,Y)
        rm_dir_if_exists(data_root / "processed")
