from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from stgym.data_loader.mouse_preoptic import MousePreopticDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples
    data = {
        "Animal_ID": ["animal1", "animal1", "animal1", "animal2", "animal2", "animal2"],
        "Cell_ID": ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        "Centroid_X": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Centroid_Y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "Animal_sex": ["m", "m", "m", "f", "f", "f"],
        "Behavior": [
            "aggressive",
            "aggressive",
            "aggressive",
            "social",
            "social",
            "social",
        ],
        "Cell_class": ["class1", "class2", "class1", "class1", "class2", "class1"],
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "nan_feature_to_drop": [np.nan]
        * 6,  # this feature is added to mimic the real scenario of this dataset
    }
    return pd.DataFrame(data)


def test_mouse_preoptic_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/mouse-preoptic")
        rm_dir_if_exists(data_root / "processed")
        ds = MousePreopticDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples

        assert ds.y.min() == 0
        assert ds.y.max() == 1

        for data in ds:
            assert data.x.shape == (3, 3)  # 3 cells, 3 features
            assert data.y.shape == (1,)  # single label per graph
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)
