from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .base import AbstractDataset

ID_COL = "barcode"
GROUP_COLS = ["sample_id"]
POS_COLS = ["array_x", "array_y"]
LABEL_COL = "cancer_type"
# Label mapping: STAD -> 0, BLCA -> 1
LABEL_MAPPING = {"STAD": 0, "BLCA": 1}
# Metadata columns that are not gene expression features
METADATA_COLS = ["x1", "x2", "x3", "x4", "x5", "pixel_x", "pixel_y"]
NON_FEATURE_COLS = [ID_COL] + GROUP_COLS + POS_COLS + [LABEL_COL] + METADATA_COLS

N_TOP_GENES = 2000

RAW_FILE_NAME = "source.csv"


class GastricBladderCancerDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        """
        Process the gastric-bladder-cancer dataset for graph classification.

        Each sample (patient) becomes a graph where:
        - Nodes are spatial spots/cells
        - Node features are gene expression values
        - Node positions are spatial coordinates
        - Graph label is cancer type (STAD vs BLCA)
        """
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)

        # drop columns with NaN values
        df.dropna(axis=1, how="any", inplace=True)

        # Extract gene expression columns (exclude metadata)
        feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]

        # Select top N highly variable genes by variance across all cells
        feature_cols = select_hvg(df[feature_cols], N_TOP_GENES)

        # Group by sample_id to create separate graphs for each patient
        groups = list(df.groupby(GROUP_COLS))
        data_list = []

        for (sample_id,), sample_df in groups:
            # Extract spatial positions
            pos = torch.Tensor(sample_df[POS_COLS].values)

            # Extract label and convert to integer using mapping
            cancer_type = sample_df[LABEL_COL].unique()
            assert len(cancer_type) == 1, (
                f"Sample {sample_id} has multiple labels: {cancer_type}"
            )

            cancer_type = cancer_type[0]
            assert cancer_type in LABEL_MAPPING, f"Unknown cancer type: {cancer_type}"

            y = torch.tensor(LABEL_MAPPING[cancer_type])

            x = torch.Tensor(sample_df[feature_cols].values)

            # Validate data shapes
            assert x.shape[0] == pos.shape[0], (
                f"Feature and position shape mismatch for {sample_id}"
            )
            assert pos.shape[1] == 2, f"Position should be 2D for {sample_id}"

            data_list.append(Data(x=x, y=y, pos=pos))

        print(f"Processed {len(data_list)} samples:")
        for i, data in enumerate(data_list):
            print(
                f"  Sample {i}: {data.x.shape[0]} cells, {data.x.shape[1]} features, label={data.y.item()}"
            )

        return data_list


def select_hvg(gene_df: pd.DataFrame, n_top: int) -> list[str]:
    """Select top n_top highly variable genes by variance."""
    variances = gene_df.var()
    top_genes = variances.nlargest(n_top).index.tolist()
    return top_genes
