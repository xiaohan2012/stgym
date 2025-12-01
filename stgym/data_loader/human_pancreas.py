from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

RAW_FILE_NAME = "source.csv"
LABEL_COL = "cell_type"
GROUP_COLS = ["sample_id"]
POS_COLS = ["x_coord", "y_coord"]
COLUMNS_TO_DROP = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
    "stage",
    "section",
]


class HumanPancreasDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # Load the preprocessed human pancreas development dataset
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)

        # Convert cell type labels to categorical codes
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes

        # Group by sample ID to create separate graphs for each sample
        groups = list(df.groupby(GROUP_COLS))
        data_list = []

        for name, sample_df in groups:
            # Extract spatial coordinates
            pos = torch.Tensor(sample_df[POS_COLS].values)

            # Extract labels
            y = torch.Tensor(sample_df[LABEL_COL].values).type(torch.int32)

            # Extract features (gene expression + cell type proportions)
            feature_cols = [
                col
                for col in sample_df.columns
                if col.startswith(("gene_", "prop_"))
                and col not in COLUMNS_TO_DROP + GROUP_COLS + POS_COLS + [LABEL_COL]
            ]

            if len(feature_cols) == 0:
                raise ValueError(
                    "No feature columns found with 'gene_' or 'prop_' prefix"
                )

            features = sample_df[feature_cols].values
            x = torch.Tensor(features)

            # Verify dimensions match
            assert (
                x.shape[0] == y.shape[0] == pos.shape[0]
            ), f"Shape mismatch: x={x.shape}, y={y.shape}, pos={pos.shape}"

            # Create PyTorch Geometric Data object
            data_list.append(Data(x=x, y=y, pos=pos))

        return data_list
