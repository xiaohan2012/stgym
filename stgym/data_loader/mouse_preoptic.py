from pathlib import Path

import pandas as pd
import torch
from logzero import logger
from torch_geometric.data import Data

from .base import AbstractDataset

ID_COL = "Cell_ID"
GROUP_COLS = ["Animal_ID"]

POS_COLS = ["Centroid_X", "Centroid_Y"]
LABEL_COL = "Behavior"
COLS_TO_DROP = ["Cell_class"]
RAW_FILE_NAME = "source.csv"


class MousePreopticDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # from: ~/Desktop/Codex数据集-2025.3.26/dataset2/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes
        df["Animal_sex"] = pd.Categorical(df["Animal_sex"]).codes
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
