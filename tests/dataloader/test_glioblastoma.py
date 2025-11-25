from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from stgym.data_loader.glioblastoma import GlioblastomaDataset
from stgym.utils import rm_dir_if_exists


@patch("pandas.read_csv")
class TestGlioblastomaDataset:
    """Test GlioblastomaDataset class."""

    @property
    def base_gene_expr_data(self) -> Dict[str, Any]:
        """Base gene expression test data with 3 samples, 2 spots each, 10 genes."""
        gene_names = [f"GENE_{i}" for i in range(10)]

        data = {
            "barcode": [
                "BARCODE001-1",
                "BARCODE002-1",  # Sample 1
                "BARCODE003-1",
                "BARCODE004-1",  # Sample 2
                "BARCODE005-1",
                "BARCODE006-1",  # Sample 3
            ],
            "in_tissue": [1] * 6,
            "array_row": [10, 11, 20, 21, 30, 31],
            "array_col": [10, 11, 20, 21, 30, 31],
            "x_coord": [1000.0, 1100.0, 2000.0, 2100.0, 3000.0, 3100.0],
            "y_coord": [1000.0, 1100.0, 2000.0, 2100.0, 3000.0, 3100.0],
            "sample_id": [
                "#UKF304_T_ST",
                "#UKF304_T_ST",  # tumor sample
                "#UKF334_C_ST",
                "#UKF334_C_ST",  # cortex sample
                "#UKF256_TC_ST",
                "#UKF256_TC_ST",  # tumor_core sample
            ],
            "patient_id": ["UKF304", "UKF304", "UKF334", "UKF334", "UKF256", "UKF256"],
            "tissue_type": [
                "tumor",
                "tumor",
                "cortex",
                "cortex",
                "tumor",
                "tumor",
            ],
        }

        # Add synthetic gene expression data (log-normalized)
        np.random.seed(42)  # Reproducible
        for gene in gene_names:
            # Different expression patterns for different tissue types
            expr_values = []
            for tissue in data["tissue_type"]:
                if tissue == "tumor":
                    expr_values.append(
                        np.random.lognormal(2.0, 1.0)
                    )  # Higher expression
                elif tissue == "cortex":
                    expr_values.append(
                        np.random.lognormal(1.0, 0.5)
                    )  # Medium expression
                else:  # tumor (includes former tumor_core)
                    expr_values.append(
                        np.random.lognormal(1.5, 0.8)
                    )  # Variable expression
            data[gene] = expr_values

        return data

    @property
    def known_values_data(self) -> Dict[str, Any]:
        """Test data with known gene expression values for validation."""
        return {
            "barcode": ["BARCODE001-1", "BARCODE002-1"],
            "in_tissue": [1, 1],
            "array_row": [10, 11],
            "array_col": [10, 11],
            "x_coord": [1000.0, 1100.0],
            "y_coord": [1000.0, 1100.0],
            "sample_id": ["#TEST_SAMPLE", "#TEST_SAMPLE"],
            "patient_id": ["TEST", "TEST"],
            "tissue_type": ["tumor", "tumor"],
            "EGFR": [3.5, 4.2],  # High expression
            "TP53": [0.1, 0.3],  # Low expression
            "PCNA": [2.1, 2.8],  # Medium expression
        }

    @property
    def metadata_filtering_data(self) -> Dict[str, Any]:
        """Test data to verify metadata columns are properly excluded from features."""
        return {
            "barcode": ["BARCODE001-1"],
            "in_tissue": [1],
            "array_row": [10],
            "array_col": [10],
            "x_coord": [1000.0],
            "y_coord": [1000.0],
            "sample_id": ["#TEST_SAMPLE"],
            "patient_id": ["TEST"],
            "tissue_type": ["tumor"],
            "ACTUAL_GENE": [2.5],  # Only this should be a feature
        }

    @property
    def inconsistent_labels_data(self) -> Dict[str, Any]:
        """Test data with inconsistent labels within the same sample (should fail)."""
        return {
            "barcode": ["BARCODE001-1", "BARCODE002-1"],
            "in_tissue": [1, 1],
            "array_row": [10, 11],
            "array_col": [10, 11],
            "x_coord": [1000.0, 1100.0],
            "y_coord": [1000.0, 1100.0],
            "sample_id": ["#UKF304_T_ST", "#UKF304_T_ST"],  # Same sample
            "patient_id": ["UKF304", "UKF304"],
            "tissue_type": ["tumor", "cortex"],  # Different labels - should fail
            "GENE_0": [1.5, 2.0],
            "GENE_1": [0.8, 1.2],
        }

    def _cleanup_test_dir(self, data_root: Path) -> None:
        """Helper to clean up test directories."""
        rm_dir_if_exists(data_root / "processed")

    def test_basic(self, mock_read_csv) -> None:
        """Test basic dataset loading and structure validation."""
        mock_read_csv.return_value = pd.DataFrame(self.base_gene_expr_data)
        data_root = Path("./tests/data/glioblastoma")
        self._cleanup_test_dir(data_root)

        ds = GlioblastomaDataset(root=data_root)

        # Dataset structure validation
        assert len(ds) == 3  # 3 unique sample_ids
        assert ds.num_classes == 2  # cortex, tumor (binary classification)
        assert ds.num_features == 10  # 10 genes in mock data

        # Label encoding validation (alphabetical: cortex=0, tumor=1)
        labels = [data.y.item() for data in ds]
        assert set(labels) == {0, 1}

        # Sample structure validation
        for data in ds:
            assert data.x.shape == (2, 10)  # 2 spots, 10 genes
            assert data.pos.shape == (2, 2)  # 2D positions
            assert data.y.dtype == torch.long
            assert data.x.dtype == torch.float
            assert data.pos.dtype == torch.float

            # Data quality validation
            assert not torch.isnan(data.x).any()
            assert not torch.isinf(data.x).any()

            # Metadata attributes validation
            assert hasattr(data, "sample_id")
            assert hasattr(data, "patient_id")
            assert hasattr(data, "num_spots")
            assert hasattr(data, "num_genes")
            assert data.num_spots == 2
            assert data.num_genes == 10

        self._cleanup_test_dir(data_root)

    def test_feature_extraction_accuracy(self, mock_read_csv) -> None:
        """Test that gene expression features are correctly extracted with known values."""
        mock_read_csv.return_value = pd.DataFrame(self.known_values_data)
        data_root = Path("./tests/data/glioblastoma")
        self._cleanup_test_dir(data_root)

        ds = GlioblastomaDataset(root=data_root)

        # Should have 1 graph with 2 spots and 3 gene features
        assert len(ds) == 1
        data = ds[0]
        assert data.x.shape == (2, 3)

        # Validate exact gene expression values
        expected_features = torch.tensor(
            [[3.5, 0.1, 2.1], [4.2, 0.3, 2.8]], dtype=torch.float
        )
        torch.testing.assert_close(data.x, expected_features)

        # Validate positions
        expected_pos = torch.tensor(
            [[1000.0, 1000.0], [1100.0, 1100.0]], dtype=torch.float
        )
        torch.testing.assert_close(data.pos, expected_pos)

        self._cleanup_test_dir(data_root)

    def test_metadata_column_filtering(self, mock_read_csv) -> None:
        """Test that metadata columns are properly excluded from features."""
        mock_read_csv.return_value = pd.DataFrame(self.metadata_filtering_data)
        data_root = Path("./tests/data/glioblastoma")
        self._cleanup_test_dir(data_root)

        ds = GlioblastomaDataset(root=data_root)

        # Should have only 1 feature (the actual gene)
        data = ds[0]
        assert data.x.shape == (1, 1)
        assert data.x[0, 0] == pytest.approx(2.5)

        self._cleanup_test_dir(data_root)

    def test_label_consistency_validation(self, mock_read_csv) -> None:
        """Test that samples with inconsistent tissue type labels are rejected."""
        mock_read_csv.return_value = pd.DataFrame(self.inconsistent_labels_data)
        data_root = Path("./tests/data/glioblastoma")
        self._cleanup_test_dir(data_root)

        # Should raise an assertion error due to multiple labels per sample
        with pytest.raises(AssertionError, match="multiple labels"):
            ds = GlioblastomaDataset(root=data_root)
            _ = list(ds)  # Force processing

        self._cleanup_test_dir(data_root)
