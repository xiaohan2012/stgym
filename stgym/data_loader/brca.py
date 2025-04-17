from pathlib import Path
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  # , download_url

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


class BRCADataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        # raise NotImplementedError("Download not supported yet!")
        pass

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
                sample_df.drop(columns=[ID_COL] + GROUP_COLS + POS_COLS).values
            )

            data_list.append(Data(x=x, y=y, pos=pos))
        return data_list

    def process(self):
        data_list = self.process_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        return data_list
