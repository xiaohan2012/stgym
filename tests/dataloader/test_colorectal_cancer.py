from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from stgym.data_loader.colorectal_cancer import ColorectalCancerDataset
from stgym.utils import rm_dir_if_exists


@pytest.fixture
def mock_df():
    # Create a toy dataframe with the required columns for colorectal cancer dataset
    data = {
        "Barcode": ["AAAC1", "AAAC2", "AAAC3", "AAAC4", "AAAC5", "AAAC6"],
        "UMAP1": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "UMAP2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "Patient": ["P1CRC", "P1CRC", "P1CRC", "P2NAT", "P2NAT", "P2NAT"],
        "BC": ["BC1", "BC1", "BC1", "BC2", "BC2", "BC2"],
        "Level2": ["NK", "CD4 T cell", "NK", "Plasma", "CD4 T cell", "Plasma"],
        "C1QA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "FGR": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "CSF3R": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    }
    return pd.DataFrame(data)


def test_colorectal_cancer_dataset(mock_df):
    with patch("pandas.read_csv", return_value=mock_df):
        data_root = Path("./tests/data/colorectal_cancer")
        rm_dir_if_exists(data_root / "processed")
        ds = ColorectalCancerDataset(root=data_root)
        assert len(ds) == 2  # there are two graph samples (P1CRC, P2NAT)

        # Test all patients
        for i in range(len(ds)):
            data = ds[i]
            assert data.x.shape == (3, 3)  # 3 cells, 3 gene features
            assert data.y.shape == (3,)  # 3 labels
            assert data.y.dtype == torch.long
            assert data.pos.shape == (3, 2)  # 3 positions (UMAP1, UMAP2)

        # Check that labels are properly encoded as categorical codes
        assert ds.y.min() >= 0  # categorical codes start from 0

        rm_dir_if_exists(data_root / "processed")
