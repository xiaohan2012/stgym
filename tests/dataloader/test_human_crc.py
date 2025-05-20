from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.human_crc import HumanCRCDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        "CellID": ["cell1", "cell2", "cell3"],
        "Region": ["region1", "region1", "region1"],
        "ClusterID": [1, 1, 1],
        "ClusterName": ["cluster1", "cluster1", "cluster1"],
        "neighborhood name": ["n1", "n1", "n1"],
        "patients": ["patient1", "patient1", "patient1"],
        "Z:Z": [0, 0, 0],
        "X:X": [1.0, 2.0, 3.0],
        "Y:Y": [1.0, 2.0, 3.0],
        "neighborhood10": [0, 1, 0],
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6],
    }
    return pd.DataFrame(data)


def test_human_crc_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/human-crc")
        rm_dir_if_exists(data_root / "processed")
        ds = HumanCRCDataset(root=data_root)
        assert len(ds) == 1  # there is only one graph sample
        data = ds[0]
        assert data.x.shape == (3, 2)  # 3 cells, 2 features
        assert data.y.shape == (3,)  # 3 labels
        assert data.pos.shape == (3, 2)  # 3 positions (x,y)
