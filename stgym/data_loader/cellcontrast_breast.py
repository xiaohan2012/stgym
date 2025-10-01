from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

RAW_FILE_NAME = "source.csv"  # this is fixed
LABEL_COL = "cell_type"  # ground truth label column
GROUP_COLS = ["sample_uuid"]  # sample identification column
POS_COLS = ["UMAP_1", "UMAP_2"]  # spatial coordinates
COLUMNS_TO_DROP = [
    "cellid",
    "donor_id",
    "self_reported_ethnicity",
    "development_stage",
]  # misc. information not relevant to the ML task


class CellcontrastBreastDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # CellContrast breast cancer spatial transcriptomics dataset
        # Source: https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            pos = torch.Tensor(sample_df[POS_COLS].values)
            y = torch.Tensor(sample_df[LABEL_COL].values).type(torch.long)
            features = sample_df.drop(
                columns=COLUMNS_TO_DROP + GROUP_COLS + POS_COLS + [LABEL_COL]
            ).values

            x = torch.Tensor(features)

            assert x.shape[0] == y.shape[0] == pos.shape[0]
            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
