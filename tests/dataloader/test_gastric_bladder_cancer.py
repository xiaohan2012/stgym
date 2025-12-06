from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.gastric_bladder_cancer import GastricBladderCancerDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns for gastric-bladder-cancer dataset
    data = {
        # ID_COL
        "barcode": [
            "AAACAGAGCGACTCCT-1",
            "AAACCGGGTAGGTACC-1",
            "AAACCGTTCGTCCAGG-1",
            "BBACAGAGCGACTCCT-1",
            "BBACCGGGTAGGTACC-1",
            "BBACCGTTCGTCCAGG-1",
        ],
        # Spatial metadata columns (not used as features)
        "x1": [1, 1, 1, 1, 1, 1],
        "x2": [14, 42, 52, 15, 43, 53],
        "x3": [94, 28, 42, 95, 29, 43],
        "x4": [6412, 2661, 3387, 6413, 2662, 3388],
        "x5": [6897, 4384, 3393, 6898, 4385, 3394],
        # POS_COLS (spatial coordinates)
        "array_x": [14, 42, 52, 15, 43, 53],
        "array_y": [94, 28, 42, 95, 29, 43],
        "pixel_x": [6412, 2661, 3387, 6413, 2662, 3388],
        "pixel_y": [6897, 4384, 3393, 6898, 4385, 3394],
        # GROUP_COLS (sample grouping)
        "sample_id": ["STAD-G1", "STAD-G1", "STAD-G1", "BLCA-B1", "BLCA-B1", "BLCA-B1"],
        # LABEL_COL (cancer type)
        "cancer_type": ["STAD", "STAD", "STAD", "BLCA", "BLCA", "BLCA"],
        # Gene expression features
        "SAMD11": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "NOC2L": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "KLHL17": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }
    return pd.DataFrame(data)


def test_gastric_bladder_cancer_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/gastric-bladder-cancer-test")
        ds = GastricBladderCancerDataset(root=data_root)

        # Should have 2 graphs (2 samples: STAD-G1 and BLCA-B1)
        assert len(ds) == 2, f"There should be 2 graph samples but got {len(ds)}"

        # Note: Due to groupby ordering, BLCA samples may come first
        # Test samples - we have 2 samples with 3 cells each and 10 features
        # (3 gene features + 7 metadata columns that become features)
        data_0 = ds[0]
        assert data_0.x.shape == (3, 10), "Shape of x should be (3 cells, 10 features)"
        assert data_0.y.item() in [0, 1], "Label should be 0 (STAD) or 1 (BLCA)"
        assert data_0.pos.shape == (
            3,
            2,
        ), "Shape of pos should be (3 positions, 2 coordinates)"

        # Test second sample
        data_1 = ds[1]
        assert data_1.x.shape == (3, 10), "Shape of x should be (3 cells, 10 features)"
        assert data_1.y.item() in [0, 1], "Label should be 0 (STAD) or 1 (BLCA)"
        assert data_1.pos.shape == (
            3,
            2,
        ), "Shape of pos should be (3 positions, 2 coordinates)"

        # Verify we have both label types
        labels = [ds[i].y.item() for i in range(len(ds))]
        assert set(labels) == {0, 1}, "Should have both STAD (0) and BLCA (1) labels"

        # Verify spatial coordinates are properly extracted (just check structure, not exact values)
        for i in range(2):
            data = ds[i]
            assert data.pos.min() >= 0, "Positions should be non-negative"
            assert data.pos.shape[1] == 2, "Positions should be 2D coordinates"

        # Clean up
        rm_dir_if_exists(data_root / "processed")


def test_single_label_validation():
    """Test that the dataset validates single label per sample."""
    # Create invalid data with mixed labels in one sample
    invalid_data = {
        "barcode": ["cell1", "cell2", "cell3"],
        "array_x": [1, 2, 3],
        "array_y": [1, 2, 3],
        "sample_id": ["SAMPLE-1", "SAMPLE-1", "SAMPLE-1"],
        "cancer_type": ["STAD", "BLCA", "STAD"],  # Mixed labels - should fail
        "GENE1": [0.1, 0.2, 0.3],
    }
    invalid_df = pd.DataFrame(invalid_data)

    with patch("pandas.read_csv", return_value=invalid_df):
        data_root = Path("./tests/data/gastric-bladder-cancer-invalid")

        # Should raise assertion error during dataset creation due to multiple labels per sample
        with pytest.raises(AssertionError, match="multiple labels"):
            ds = GastricBladderCancerDataset(root=data_root)

        rm_dir_if_exists(data_root / "processed")


def test_unknown_cancer_type():
    """Test handling of unknown cancer types."""
    # Create data with unknown cancer type
    invalid_data = {
        "barcode": ["cell1", "cell2"],
        "array_x": [1, 2],
        "array_y": [1, 2],
        "sample_id": ["SAMPLE-1", "SAMPLE-1"],
        "cancer_type": ["UNKNOWN", "UNKNOWN"],  # Unknown type - should fail
        "GENE1": [0.1, 0.2],
    }
    invalid_df = pd.DataFrame(invalid_data)

    with patch("pandas.read_csv", return_value=invalid_df):
        data_root = Path("./tests/data/gastric-bladder-cancer-unknown")

        # Should raise assertion error during dataset creation due to unknown cancer type
        with pytest.raises(AssertionError, match="Unknown cancer type"):
            ds = GastricBladderCancerDataset(root=data_root)

        rm_dir_if_exists(data_root / "processed")
