from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from stgym.data_loader.mouse_kidney import MouseKidneyDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples
    data = {
        "Source-Sample-ID": [
            "GSM5713332",
            "GSM5713332",
            "GSM5713332",
            "GSM5713333",
            "GSM5713333",
            "GSM5713333",
        ],
        "Sample_title": [
            "BTBR-wt/wt-1a",
            "BTBR-wt/wt-1a",
            "BTBR-wt/wt-1a",
            "BTBR-ob/ob-1a",
            "BTBR-ob/ob-1a",
            "BTBR-ob/ob-1a",
        ],
        "barcodes": [
            "AAAAACCTCGGTAC",
            "AAAAAGCTCTAAAG",
            "AAAAATACGCATAT",
            "BBBBACCTCGGTAC",
            "BBBBAGCTCTAAAG",
            "BBBBATACGCATAT",
        ],
        "xcoord": [4802.0, 5055.7, 4123.5, 3456.7, 4123.8, 3789.2],
        "ycoord": [2201.6, 3087.5, 2876.3, 2134.5, 2567.8, 2890.1],
        "Disease-Status": [
            "Control",
            "Control",
            "Control",
            "Diabetes",
            "Diabetes",
            "Diabetes",
        ],
        "cell_type": ["PCT_1", "PCT_1", "DCT", "PCT_2", "PCT_2", "DCT"],
        "Tissue": ["Kidney Cross-Section"] * 6,
        "Age": ["13 week"] * 6,
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "feature3": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "nan_feature_to_drop": [np.nan]
        * 6,  # this feature is added to mimic the real scenario of this dataset
    }
    return pd.DataFrame(data)


def test_mouse_kidney_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/mouse-kidney")
        rm_dir_if_exists(data_root / "processed")
        ds = MouseKidneyDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples

        assert ds.y.min() == 0
        assert ds.y.max() == 1

        for data in ds:
            assert data.x.shape == (
                3,
                3,
            )  # 3 cells, 3 features (after dropping nan feature)
            assert data.y.shape == (1,)  # single label per graph
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (x,y)
        rm_dir_if_exists(data_root / "processed")
