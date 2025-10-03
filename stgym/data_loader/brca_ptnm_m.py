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
    "PTNM_M",  # consider removing this column
    "PTNM_T",
    "PTNM_N",
    "Post-surgeryTx",
    "clinical_type",
]

POS_COLS = ["X_centroid", "Y_centroid"]
LABEL_COL = "PTNM_M"

RAW_FILE_NAME = "source.csv"


class BRCAPTNMMDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            # Verify that each sample has only one unique label
            unique_labels = sample_df[LABEL_COL].unique()
            assert (
                len(unique_labels) == 1
            ), f"Sample has inconsistent labels: {unique_labels}"

            pos = torch.Tensor(sample_df[POS_COLS].values)
            y = torch.tensor(unique_labels[0], dtype=torch.long)

            x = torch.Tensor(
                sample_df.drop(columns=[ID_COL] + GROUP_COLS + POS_COLS).values
            )

            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
