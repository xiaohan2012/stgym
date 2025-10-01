from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.charville import CharvilleDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        "CELL_ID": [1, 2, 3, 4, 5, 6],
        "X": [1371, 1390, 850, 884, 1315, 1340],
        "Y": [3, 6, 8, 7, 8, 7],
        "SIZE": [0.125, 0.342, 0.564, 0.379, 0.520, 0.479],
        "CELL_TYPE": [
            "Tumor 2 (Ki67 Proliferating)",
            "Tumor 3",
            "Other",
            "CD4 T cell",
            "Tumor 3",
            "Tumor 3",
        ],
        "reg_id": [
            "Charville_c001_v001_r001_reg001",
            "Charville_c001_v001_r001_reg001",
            "Charville_c001_v001_r001_reg001",
            "Charville_c001_v001_r001_reg002",
            "Charville_c001_v001_r001_reg002",
            "Charville_c001_v001_r001_reg002",
        ],
        "CD107a": [0.972, 1.196, -0.192, -0.247, -0.070, 0.683],
        "CD117": [-0.416, -0.681, -0.459, -0.131, -0.096, -0.095],
        "CD11b": [1.397, 0.559, 1.243, 3.042, -0.346, -0.524],
        "CD11c": [-0.128, -0.405, -0.535, 3.345, 0.324, 0.283],
    }
    return pd.DataFrame(data)


def test_charville_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/charville")
        rm_dir_if_exists(data_root / "processed")
        ds = CharvilleDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples
        # Test all regions
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (
                3,
                4,
            )  # 3 cells, 4 features (CD107a, CD117, CD11b, CD11c)
            assert data.y.shape == (3,)  # 3 labels
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (X,Y)
        rm_dir_if_exists(data_root / "processed")
