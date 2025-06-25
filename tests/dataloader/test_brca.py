from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.brca import BRCADataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns
    data = {
        # ID_COL
        "cellID": ["cell1", "cell2", "cell3"],
        # GROUP_COLS
        "grade": ["1", "1", "1"],
        "gender": ["f", "f", "f"],
        "age": [50, 50, 50],
        "Patientstatus": ["alive", "alive", "alive"],
        "diseasestatus": ["tumor", "tumor", "tumor"],
        "PTNM_M": ["m", "m", "m"],
        "PTNM_T": ["t", "t", "t"],
        "PTNM_N": ["n", "n", "n"],
        "Post-surgeryTx": ["no", "no", "no"],
        "clinical_type": ["a", "a", "a"],
        # POS_COLS
        "X_centroid": [1.0, 2.0, 3.0],
        "Y_centroid": [1.0, 2.0, 3.0],
        # Feature columns
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6],
    }
    return pd.DataFrame(data)


def test_brca_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/brca-test")
        rm_dir_if_exists(data_root / "processed")
        ds = BRCADataset(root=data_root)
        assert len(ds) == 1, "There should be only one graph sample"
        data = ds[0]
        assert data.x.shape == (3, 2), "Shape of x should be (3 cells, 2 features)"
        assert data.y.item() == 1, "Label should be 1 (tumor)"
        assert data.pos.shape == (3, 2), "Shape of pos should be (3 positions (x,y))"

        rm_dir_if_exists(data_root / "processed")
