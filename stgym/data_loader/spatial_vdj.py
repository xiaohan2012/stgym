from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

# Column definitions for SpatialVDJ dataset
ID_COL = "spatial_bc"
GROUP_COLS = ["sample_id"]  # Each sample represents a tissue section
POS_COLS = ["x", "y"]  # Spatial coordinates
LABEL_COL = "tissue_type"  # Cancer vs normal classification
POSITIVE_LABEL = "cancer"  # Cancer tissue label

# Feature columns - B/T cell receptor and cell type information
FEATURE_COLS = [
    "B.Cells",
    "FDC",
    "Myeloid.Cells",
    "Plasmablast",
    "T.Cells",
    "TR_UMI_count",
    "IG_UMI_count",
    "TRA_UMI_count",
    "TRB_UMI_count",
    "IGK_UMI_count",
    "IGL_UMI_count",
    "IGH_UMI_count",
    "IGHA_UMI_count",
    "IGHG_UMI_count",
    "IGHM_UMI_count",
    "IGHD_UMI_count",
]

RAW_FILE_NAME = "source.csv"


class SpatialVDJDataset(AbstractDataset):
    """
    SpatialVDJ dataset for cancer vs normal tissue classification.

    This dataset combines:
    - Breast cancer tissue sections (2 patients, multiple regions each)
    - Normal tonsil tissue sections (6 sections)

    Each sample represents a tissue section that can be classified as:
    - Cancer (breast tumor tissue)
    - Normal (tonsil lymphoid tissue)

    Features include cell type compositions and B/T cell receptor counts
    providing rich immunological context for spatial analysis.
    """

    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)

        # Ensure all feature columns exist, fill missing with zeros
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # Group by sample_id (each sample is a tissue section)
        groups = list(df.groupby(GROUP_COLS))
        data_list = []

        for name, sample_df in groups:
            sample_id = name[0] if isinstance(name, tuple) else name

            # Get spatial coordinates
            pos = torch.tensor(sample_df[POS_COLS].values, dtype=torch.float)

            # Get binary label: cancer=1, normal=0
            tissue_types = sample_df[LABEL_COL].unique()
            if len(tissue_types) != 1:
                raise ValueError(
                    f"Sample {sample_id} has multiple tissue types: {tissue_types}"
                )

            tissue_type = tissue_types[0]
            y = torch.tensor(
                1 if tissue_type == POSITIVE_LABEL else 0, dtype=torch.long
            )

            # Get feature matrix (cell composition + immune receptor counts)
            feature_data = sample_df[FEATURE_COLS].values
            x = torch.tensor(feature_data, dtype=torch.float)

            # Validate data shapes
            assert len(pos) == len(
                x
            ), f"Position and feature matrices have different lengths: {len(pos)} vs {len(x)}"
            assert x.shape[1] == len(
                FEATURE_COLS
            ), f"Feature matrix has wrong number of columns: {x.shape[1]} vs {len(FEATURE_COLS)}"

            # Create graph data object
            data = Data(x=x, y=y, pos=pos)
            data.sample_id = sample_id
            data.tissue_type = tissue_type
            data.num_spots = len(x)

            data_list.append(data)

        print(f"Processed {len(data_list)} samples from SpatialVDJ dataset")

        # Print dataset summary
        cancer_samples = sum(1 for data in data_list if data.y.item() == 1)
        normal_samples = sum(1 for data in data_list if data.y.item() == 0)
        total_spots = sum(data.num_spots for data in data_list)

        print(f"Dataset summary:")
        print(f"  Cancer samples: {cancer_samples}")
        print(f"  Normal samples: {normal_samples}")
        print(f"  Total spots: {total_spots}")
        print(f"  Average spots per sample: {total_spots / len(data_list):.1f}")

        return data_list
