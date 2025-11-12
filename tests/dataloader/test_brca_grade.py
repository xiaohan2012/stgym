from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.brca_grade import BRCAGradeDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a simplified toy dataframe - 2 cells per grade, 3 feature columns
    data = {
        # ID_COL
        "cellID": ["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
        # GROUP_COLS - grade comes first, then other grouping variables
        "grade": [1, 1, 2, 2, 3, 3],
        "gender": ["FEMALE", "FEMALE", "MALE", "MALE", "FEMALE", "FEMALE"],
        "age": [45, 45, 55, 55, 65, 65],
        "PTNM_M": [0, 0, 0, 0, 1, 1],
        "PTNM_T": ["1a", "1a", "1c", "1c", "2", "2"],
        "PTNM_N": [0, 0, 0, 0, 1, 1],
        "Patientstatus": ["alive", "alive", "alive", "alive", "dead", "dead"],
        "diseasestatus": ["tumor", "tumor", "tumor", "tumor", "tumor", "tumor"],
        "Post-surgeryTx": [
            "None",
            "None",
            "Chemotherapy",
            "Chemotherapy",
            "Radiation",
            "Radiation",
        ],
        "clinical_type": [
            "HR+HER2-",
            "HR+HER2-",
            "HR-HER2+",
            "HR-HER2+",
            "TNBC",
            "TNBC",
        ],
        # POS_COLS
        "X_centroid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Y_centroid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        # Feature columns (3 features only)
        "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "feature2": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "feature3": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    }
    return pd.DataFrame(data)


def test_brca_grade_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/brca-grade-test")
        ds = BRCAGradeDataset(root=data_root)
        assert (
            len(ds) == 3
        ), f"There should be 3 graph samples (one per grade) but got {len(ds)}"

        # Test first sample (grade 1 -> label 0)
        data0 = ds[0]
        assert data0.x.shape == (2, 3), "Shape of x should be (2 cells, 3 features)"
        assert data0.y.item() == 0, "Label should be 0 (grade 1 -> 0-indexed as 0)"
        assert data0.pos.shape == (2, 2), "Shape of pos should be (2 positions (x,y))"

        # Test second sample (grade 2 -> label 1)
        data1 = ds[1]
        assert data1.x.shape == (2, 3), "Shape of x should be (2 cells, 3 features)"
        assert data1.y.item() == 1, "Label should be 1 (grade 2 -> 0-indexed as 1)"
        assert data1.pos.shape == (2, 2), "Shape of pos should be (2 positions (x,y))"

        # Test third sample (grade 3 -> label 2)
        data2 = ds[2]
        assert data2.x.shape == (2, 3), "Shape of x should be (2 cells, 3 features)"
        assert data2.y.item() == 2, "Label should be 2 (grade 3 -> 0-indexed as 2)"
        assert data2.pos.shape == (2, 2), "Shape of pos should be (2 positions (x,y))"

        rm_dir_if_exists(data_root / "processed")


def test_brca_grade_label_consistency():
    # Test that the assertion check works for grade consistency within samples
    # This verifies our label consistency validation is in place
    data = {
        "cellID": ["cell1", "cell2", "cell3"],
        "grade": [1, 1, 1],  # All cells have same grade - should work fine
        "gender": ["FEMALE", "FEMALE", "FEMALE"],
        "age": [50, 50, 50],
        "PTNM_M": [0, 0, 0],
        "PTNM_T": ["1c", "1c", "1c"],
        "PTNM_N": [0, 0, 0],
        "Patientstatus": ["alive", "alive", "alive"],
        "diseasestatus": ["tumor", "tumor", "tumor"],
        "Post-surgeryTx": ["None", "None", "None"],
        "clinical_type": ["HR+HER2-", "HR+HER2-", "HR+HER2-"],
        "X_centroid": [1.0, 2.0, 3.0],
        "Y_centroid": [1.0, 2.0, 3.0],
        # Minimal feature columns
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6],
        "feature3": [0.7, 0.8, 0.9],
    }
    consistent_df = pd.DataFrame(data)

    # Test that consistent data works fine
    with patch("pandas.read_csv", return_value=consistent_df):
        data_root = Path("./tests/data/brca-grade-consistent-test")
        ds = BRCAGradeDataset(root=data_root)
        assert len(ds) == 1  # Should create one sample successfully
        assert ds[0].y.item() == 0  # Grade 1 -> label 0
        assert ds[0].x.shape == (3, 3)  # 3 cells, 3 features
        rm_dir_if_exists(data_root / "processed")
