from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.human_lung import HumanLungDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples
    data = {
        "Source.Sample.ID": [
            "001C",
            "001C",
            "001C",
            "002C",
            "002C",
            "002C",
        ],
        "Source.Donor.ID": ["001C", "001C", "001C", "002C", "002C", "002C"],
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "CellType.MetaData": ["AM", "AM", "AM", "NK", "NK", "NK"],
        "cell_id": ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        "Disease.Status": [
            "Control",
            "Control",
            "Control",
            "Control",
            "Control",
            "Control",
        ],
        "Lung.Region": [
            "Parenchyma",
            "Parenchyma",
            "Parenchyma",
            "Parenchyma",
            "Parenchyma",
            "Parenchyma",
        ],
        "Sex.MetaData": ["M", "M", "M", "M", "M", "M"],
        "Age.MetaData": [22, 22, 22, 22, 22, 22],
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


def test_human_lung_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/human-lung")
        rm_dir_if_exists(data_root / "processed")
        ds = HumanLungDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples

        assert ds.y.min() == 0
        assert ds.y.max() == 1

        for data in ds:
            assert data.x.shape == (3, 2)  # 3 cells, 2 features
            assert data.y.shape == (3,)  # 3 labels
            assert len(set(data.y.numpy())) == 1
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)

        assert set(ds[0].y.numpy()) != set(ds[1].y.numpy())
        rm_dir_if_exists(data_root / "processed")
