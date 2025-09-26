from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

RAW_FILE_NAME = "source.csv"
LABEL_COL = "author_cell_type"
GROUP_COLS = ["sample_uuid"]
POS_COLS = ["UMAP_1", "UMAP_2"]
COLUMNS_TO_DROP = [
    "Unnamed: 0",
    "donor_id",
    "cell_type",
    "sex",
    "tissue",
    "development_stage",
]


class KidneyRenalMedullaDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # from: /Users/misc/Downloads/codex_dataset_2025.7.20/7/cortex of kidney renal medulla.csv
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            pos = torch.Tensor(sample_df[POS_COLS].values.astype(float))
            y = torch.Tensor(sample_df[LABEL_COL].values).type(torch.long)
            features_df = sample_df.drop(
                columns=COLUMNS_TO_DROP + GROUP_COLS + POS_COLS + [LABEL_COL]
            )
            # Ensure all feature columns are numeric
            features = features_df.values.astype(float)
            x = torch.Tensor(features)

            assert x.shape[0] == y.shape[0] == pos.shape[0]
            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
