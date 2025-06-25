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
        "X.X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Y.Y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
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

        assert ds.y.min() == 0
        assert ds.y.max() == 1

        for data in ds:
            assert data.x.shape == (3, 2)  # 3 cells, 2 features
            assert data.y.shape == (3,)  # 3 labels
            assert len(set(data.y.numpy())) == 1
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)

        assert set(ds[0].y.numpy()) != set(ds[1].y.numpy())
        rm_dir_if_exists(data_root / "processed")
