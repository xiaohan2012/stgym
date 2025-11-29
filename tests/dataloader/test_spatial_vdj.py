from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stgym.data_loader.spatial_vdj import SpatialVDJDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    """Create a toy dataframe with the required columns for SpatialVDJ dataset."""
    data = {
        # ID_COL
        "spatial_bc": [
            "AAAC-1",
            "AAAC-2",
            "AAAC-3",  # Cancer sample 1
            "BBBB-1",
            "BBBB-2",  # Cancer sample 2
            "CCCC-1",
            "CCCC-2",
            "CCCC-3",
            "CCCC-4",  # Normal sample
        ],
        # GROUP_COLS
        "sample_id": [
            "P1_RegC2",
            "P1_RegC2",
            "P1_RegC2",  # Cancer sample 1
            "P2_RegA1",
            "P2_RegA1",  # Cancer sample 2
            "Tonsil_section12",
            "Tonsil_section12",
            "Tonsil_section12",
            "Tonsil_section12",  # Normal sample
        ],
        # POS_COLS
        "x": [1000, 1001, 1002, 3000, 3001, 28360, 28361, 28362, 28363],
        "y": [1000, 1001, 1002, 1000, 1001, 34352, 34353, 34354, 34355],
        # LABEL_COL
        "tissue_type": [
            "cancer",
            "cancer",
            "cancer",  # Cancer samples
            "cancer",
            "cancer",
            "normal",
            "normal",
            "normal",
            "normal",  # Normal sample
        ],
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0],
        # Feature columns - B/T cell composition and receptor counts
        "B.Cells": [0.2, 0.3, 0.25, 0.1, 0.15, 0.58, 0.62, 0.55, 0.60],
        "FDC": [0.04, 0.03, 0.035, 0.02, 0.025, 0.04, 0.038, 0.042, 0.039],
        "Myeloid.Cells": [0.08, 0.09, 0.085, 0.05, 0.055, 0.082, 0.078, 0.085, 0.080],
        "Plasmablast": [0.0, 0.0, 0.0, 0.0, 0.0, 1e-13, 2e-13, 1.5e-13, 1.2e-13],
        "T.Cells": [0.68, 0.61, 0.635, 0.83, 0.8, 0.297, 0.32, 0.285, 0.31],
        "TR_UMI_count": [15, 12, 14, 20, 18, 6, 4, 8, 5],
        "IG_UMI_count": [8, 10, 9, 5, 7, 2, 2, 7, 3],
        "TRA_UMI_count": [3, 2, 3, 5, 4, 0, 0, 0, 0],
        "TRB_UMI_count": [12, 10, 11, 15, 14, 6, 4, 8, 5],
        "IGK_UMI_count": [2, 3, 2, 1, 2, 0, 1, 4, 1],
        "IGL_UMI_count": [3, 4, 4, 2, 3, 0, 0, 1, 1],
        "IGH_UMI_count": [3, 3, 3, 2, 2, 2, 1, 2, 1],
        "IGHA_UMI_count": [1, 1, 1, 0, 1, 0, 0, 1, 0],
        "IGHG_UMI_count": [1, 1, 1, 1, 0, 0, 0, 0, 0],
        "IGHM_UMI_count": [1, 1, 1, 1, 1, 1, 1, 0, 1],
        "IGHD_UMI_count": [0, 0, 0, 0, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_spatial_vdj_dataset(mock_df):
    """Test that SpatialVDJDataset correctly processes the mock data."""
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/spatial-vdj-test")
        ds = SpatialVDJDataset(root=data_root)

        # Should have 3 samples: 2 cancer + 1 normal
        assert len(ds) == 3, f"Expected 3 graph samples but got {len(ds)}"

        # Check cancer samples
        cancer_samples = [data for data in ds if data.y.item() == 1]
        normal_samples = [data for data in ds if data.y.item() == 0]

        assert (
            len(cancer_samples) == 2
        ), f"Expected 2 cancer samples but got {len(cancer_samples)}"
        assert (
            len(normal_samples) == 1
        ), f"Expected 1 normal sample but got {len(normal_samples)}"

        # Test first cancer sample (P1_RegC2)
        cancer_sample_1 = [data for data in ds if data.sample_id == "P1_RegC2"][0]
        assert cancer_sample_1.x.shape == (
            3,
            16,
        ), f"Expected shape (3, 16) but got {cancer_sample_1.x.shape}"
        assert cancer_sample_1.y.item() == 1, "Cancer sample should have label 1"
        assert cancer_sample_1.pos.shape == (
            3,
            2,
        ), f"Expected pos shape (3, 2) but got {cancer_sample_1.pos.shape}"
        assert cancer_sample_1.tissue_type == "cancer", "Tissue type should be cancer"
        assert cancer_sample_1.num_spots == 3, "Should have 3 spots"

        # Test second cancer sample (P2_RegA1)
        cancer_sample_2 = [data for data in ds if data.sample_id == "P2_RegA1"][0]
        assert cancer_sample_2.x.shape == (
            2,
            16,
        ), f"Expected shape (2, 16) but got {cancer_sample_2.x.shape}"
        assert cancer_sample_2.y.item() == 1, "Cancer sample should have label 1"
        assert cancer_sample_2.pos.shape == (
            2,
            2,
        ), f"Expected pos shape (2, 2) but got {cancer_sample_2.pos.shape}"

        # Test normal sample (Tonsil_section12)
        normal_sample = [data for data in ds if data.sample_id == "Tonsil_section12"][0]
        assert normal_sample.x.shape == (
            4,
            16,
        ), f"Expected shape (4, 16) but got {normal_sample.x.shape}"
        assert normal_sample.y.item() == 0, "Normal sample should have label 0"
        assert normal_sample.pos.shape == (
            4,
            2,
        ), f"Expected pos shape (4, 2) but got {normal_sample.pos.shape}"
        assert normal_sample.tissue_type == "normal", "Tissue type should be normal"
        assert normal_sample.num_spots == 4, "Should have 4 spots"

        # Test feature values are reasonable (non-negative)
        for data in ds:
            assert (data.x >= 0).all(), "All feature values should be non-negative"
            assert (
                data.x.shape[1] == 16
            ), f"Each sample should have 16 features, got {data.x.shape[1]}"

        # Clean up test directory
        rm_dir_if_exists(data_root / "processed")


def test_spatial_vdj_dataset_mixed_labels_error():
    """Test that dataset raises error when sample has mixed tissue types."""
    # Create dataframe with mixed labels in same sample
    bad_data = {
        "spatial_bc": ["AAAC-1", "AAAC-2"],
        "sample_id": ["mixed_sample", "mixed_sample"],  # Same sample
        "x": [1000, 1001],
        "y": [1000, 1001],
        "tissue_type": [
            "cancer",
            "normal",
        ],  # Different tissue types - should cause error
        "label": [1, 0],
        "B.Cells": [0.2, 0.3],
        "FDC": [0.04, 0.03],
        "Myeloid.Cells": [0.08, 0.09],
        "Plasmablast": [0.0, 0.0],
        "T.Cells": [0.68, 0.61],
        "TR_UMI_count": [15, 12],
        "IG_UMI_count": [8, 10],
        "TRA_UMI_count": [3, 2],
        "TRB_UMI_count": [12, 10],
        "IGK_UMI_count": [2, 3],
        "IGL_UMI_count": [3, 4],
        "IGH_UMI_count": [3, 3],
        "IGHA_UMI_count": [1, 1],
        "IGHG_UMI_count": [1, 1],
        "IGHM_UMI_count": [1, 1],
        "IGHD_UMI_count": [0, 0],
    }
    bad_df = pd.DataFrame(bad_data)

    with patch("pandas.read_csv", return_value=bad_df):
        data_root = Path("./tests/data/spatial-vdj-bad")

        # Should raise ValueError for mixed tissue types in same sample
        with pytest.raises(ValueError, match="has multiple tissue types"):
            ds = SpatialVDJDataset(root=data_root)

        # Clean up
        rm_dir_if_exists(data_root / "processed")
