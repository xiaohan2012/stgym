from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.human_intestine import HumanIntestineDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        "tissue": ["tissue1", "tissue1", "tissue1", "tissue1", "tissue1", "tissue1"],
        "donor": ["donor1", "donor1", "donor1", "donor1", "donor1", "donor1"],
        "unique_region": [
            "region1",
            "region1",
            "region1",
            "region2",
            "region2",
            "region2",
        ],
        "cell_type_A": ["type1", "type2", "type1", "type1", "type2", "type1"],
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "feature1": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
    }
    return pd.DataFrame(data)


def test_human_intestine_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/human-intestine")
        rm_dir_if_exists(data_root / "processed")
        ds = HumanIntestineDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples
        # Test all regions
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (3, 2)  # 3 cells, 2 features
            assert data.y.shape == (3,)  # 3 labels
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)
