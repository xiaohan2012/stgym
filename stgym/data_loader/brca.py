from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

ID_COL = "cellID"
GROUP_COLS = [
    "grade",
    "gender",
    "age",
    "Patientstatus",
    "diseasestatus",
    "PTNM_M",
    "PTNM_T",
    "PTNM_N",
    "Post-surgeryTx",
    "clinical_type",
]

POS_COLS = ["X_centroid", "Y_centroid"]
LABEL_COL = "diseasestatus"
POSITIVE_LABEL = "tumor"

RAW_FILE_NAME = "source.csv"


class BRCADataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # from: ~/Desktop/Codex数据集-2025.3.26/dataset6/BRCA_results_expression_combine.csv
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            pos = torch.Tensor(sample_df[POS_COLS].values)
            y = torch.tensor(
                (sample_df[LABEL_COL] == POSITIVE_LABEL).astype(int).unique()[0]
            )

            x = torch.Tensor(
                sample_df.drop(
                    columns=[ID_COL] + GROUP_COLS + POS_COLS + [LABEL_COL]
                ).values
            )

            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
