from pathlib import Path

import pandas as pd
import torch
from logzero import logger
from torch_geometric.data import Data

from .base import AbstractDataset

ID_COL = "barcodes"
GROUP_COLS = ["Source-Sample-ID", "Sample_title"]
LABEL_COL = "Disease-Status"
POS_COLS = ["xcoord", "ycoord"]
COLS_TO_DROP = ["cell_type", "Tissue", "Age"]
RAW_FILE_NAME = "GSE190094.parquet"


class MouseKidneyDataset(AbstractDataset):
    """
    Example:
    >> from stgym.data_loader.mouse_kidney import MouseKidneyDataset
    >> ds = MouseKidneyDataset(root="../data/mouse-kidney")
    """

    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_parquet(data_path)

        # Encode labels in-place
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes

        # Identify feature columns and drop NaN columns in-place
        non_feature_cols = [ID_COL] + GROUP_COLS + POS_COLS + [LABEL_COL] + COLS_TO_DROP
        feature_cols = [c for c in df.columns if c not in non_feature_cols]
        nan_cols = [c for c in feature_cols if df[c].isna().any()]
        if nan_cols:
            logger.info(f"Dropping columns containing NaN values: {nan_cols}")
            df.drop(columns=nan_cols, inplace=True)
            feature_cols = [c for c in feature_cols if c not in nan_cols]

        # Drop metadata columns no longer needed
        df.drop(columns=COLS_TO_DROP + [ID_COL], inplace=True)

        # Downcast float64 feature columns to float32 to halve memory
        float64_cols = df.select_dtypes("float64").columns
        df[float64_cols] = df[float64_cols].astype("float32")

        data_list = []
        for _, sample_df in df.groupby(GROUP_COLS):
            labels = sample_df[LABEL_COL].unique()
            assert len(labels) == 1, len(labels)
            y = torch.tensor(labels[0], dtype=torch.long)
            pos = torch.tensor(sample_df[POS_COLS].values, dtype=torch.float)
            x = torch.tensor(sample_df[feature_cols].values, dtype=torch.float)
            data_list.append(Data(x=x, y=y, pos=pos))

        return data_list
