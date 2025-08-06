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
RAW_FILE_NAME = "GSE190094.csv"


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
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path, sep=",")
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes
        groups = list(df.groupby(GROUP_COLS))
        data_list = []

        cols_to_drop = [ID_COL] + GROUP_COLS + POS_COLS + [LABEL_COL] + COLS_TO_DROP
        feat_df = df.drop(columns=cols_to_drop)
        nan_cols = feat_df.columns[feat_df.isna().any(axis=0)]
        logger.info(f"Dropping columns containing NaN values: {nan_cols}")

        for _, sample_df in groups:
            labels = set(sample_df[LABEL_COL].values)
            assert len(labels) == 1, len(labels)
            y = torch.tensor(list(labels)[0], dtype=torch.long)
            pos = torch.Tensor(sample_df[POS_COLS].values)
            x = torch.Tensor(
                sample_df.drop(columns=cols_to_drop + list(nan_cols)).values
            )

            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
