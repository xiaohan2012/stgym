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
    "PTNM_M",
    "PTNM_T",
    "PTNM_N",
    "Patientstatus",
    "diseasestatus",
    "Post-surgeryTx",
    "clinical_type",
]

POS_COLS = ["X_centroid", "Y_centroid"]
LABEL_COL = "grade"

RAW_FILE_NAME = "source.csv"


class BRCAGradeDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            pos = torch.Tensor(sample_df[POS_COLS].values)

            # Check that all cells in this sample have the same label
            unique_labels = sample_df[LABEL_COL].unique()
            if len(unique_labels) != 1:
                raise ValueError(f"Sample has multiple labels: {unique_labels}")

            # Convert grade to 0-indexed labels (1->0, 2->1, 3->2)
            y = torch.tensor(int(unique_labels[0]) - 1)

            x = torch.Tensor(
                sample_df.drop(
                    columns=[ID_COL] + GROUP_COLS + POS_COLS + [LABEL_COL]
                ).values
            )

            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
