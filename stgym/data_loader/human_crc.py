from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

RAW_FILE_NAME = "source.csv"
LABEL_COL = "neighborhood10"
GROUP_COLS = ["patients", "Z:Z"]
POS_COLS = ["X:X", "Y:Y"]
COLUMNS_TO_DROP = ["CellID", "Region", "ClusterID", "ClusterName", "neighborhood name"]


class HumanCRCDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # from: ~/Downloads/Codex数据集-2025.3.26/dataset3/CRC_clusters_neighborhoods_markers.csv
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)
        groups = list(df.groupby(GROUP_COLS))
        data_list = []
        for name, sample_df in groups:
            pos = torch.Tensor(sample_df[POS_COLS].values)
            y = torch.Tensor(sample_df[LABEL_COL].values).type(torch.int32)
            features = sample_df.drop(
                columns=COLUMNS_TO_DROP + GROUP_COLS + POS_COLS + [LABEL_COL]
            ).values

            x = torch.Tensor(features)

            assert x.shape[0] == y.shape[0] == pos.shape[0]
            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list
