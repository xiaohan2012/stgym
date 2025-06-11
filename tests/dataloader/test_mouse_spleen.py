from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.mouse_spleen import MouseSpleenDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples
    data = {
        "sample_Xtile_Ytile": [
            "sample1",
            "sample1",
            "sample1",
            "sample2",
            "sample2",
            "sample2",
        ],
        "Z.Z": [0, 0, 0, 0, 0, 0],
        "X:X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Y:Y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Imaging phenotype cluster ID": [1, 1, 1, 2, 2, 2],
        "niche cluster ID": [0, 1, 0, 0, 1, 0],
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


def test_mouse_spleen_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/mouse-spleen")
        rm_dir_if_exists(data_root / "processed")
        ds = MouseSpleenDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples

        # Test first sample
        data1 = ds[0]
        assert data1.x.shape == (3, 2)  # 3 cells, 2 features
        assert data1.y.shape == (3,)  # 3 labels
        assert data1.pos.shape == (3, 2)  # 3 positions (x,y)

        # Test second sample
        data2 = ds[1]
        assert data2.x.shape == (3, 2)  # 3 cells, 2 features
        assert data2.y.shape == (3,)  # 3 labels
        assert data2.pos.shape == (3, 2)  # 3 positions (x,y)
