from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.breast_cancer import BreastCancerDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with two samples
    data = {
        "Unnamed: 0": ["a", "b", "c", "d", "e", "f"],
        "donor_id": [
            "donor1",
            "donor1",
            "donor1",
            "donor2",
            "donor2",
            "donor2",
        ],
        "sample_uuid": [
            "sample1",
            "sample1",
            "sample1",
            "sample2",
            "sample2",
            "sample2",
        ],
        "UMAP_1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "UMAP_2": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "cell_type": ["T cell", "T cell", "T cell", "B cell", "B cell", "B cell"],
        "author_cell_type": ["T", "T", "T", "B", "B", "B"],
        "sex": ["F", "F", "F", "F", "F", "F"],
        "tissue": ["breast", "breast", "breast", "breast", "breast", "breast"],
        "development_stage": ["adult", "adult", "adult", "adult", "adult", "adult"],
        "ENSG00000187608": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "ENSG00000186827": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    return pd.DataFrame(data)


def test_breast_cancer_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/breast-cancer")
        rm_dir_if_exists(data_root / "processed")
        ds = BreastCancerDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples

        assert ds.y.min() == 0
        assert ds.y.max() == 1

        for data in ds:
            assert data.x.shape == (3, 2)  # 3 cells, 2 features (ENSG genes)
            assert data.y.shape == (3,)  # 3 labels
            assert len(set(data.y.numpy())) == 1
            assert data.pos.shape == (3, 2)  # 3 positions (UMAP_1, UMAP_2)

        assert set(ds[0].y.numpy()) != set(ds[1].y.numpy())
        rm_dir_if_exists(data_root / "processed")
