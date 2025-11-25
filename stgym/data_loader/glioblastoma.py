from pathlib import Path

import pandas as pd
import torch
from logzero import logger
from torch_geometric.data import Data

from .base import AbstractDataset

# Column definitions based on the CSV structure
GROUP_COLS = ["sample_id", "patient_id"]  # Use sample_id to group spots into graphs

POS_COLS = ["x_coord", "y_coord"]
LABEL_COL = "tissue_type"

# Metadata columns to exclude from features (includes 10X Visium spatial info)
METADATA_COLS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "x_coord",
    "y_coord",
    "sample_id",
    "patient_id",
    "tissue_type",
]

RAW_FILE_NAME = "source.csv"


class GlioblastomaDataset(AbstractDataset):
    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        """
        Process the glioblastoma spatial transcriptomics data with gene expression features.

        Each graph represents one spatial transcriptomics sample (e.g., #UKF304_T_ST).
        Nodes are spatial spots with gene expression features.
        Graph-level labels are tissue types (cortex, tumor, tumor_core, tumor_infiltration).

        This version uses raw gene expression data extracted from 10X Visium matrices
        instead of the processed cell type scores, providing:
        - Complete dataset utilization (all samples)
        - Rich molecular information (gene expression profiles)
        - Uniform feature space across samples
        - Standard spatial transcriptomics workflow

        Data processing steps:
        1. Load consolidated gene expression dataset created by preprocessing script
        2. Extract gene expression features (log-normalized counts)
        3. Group spots by sample_id to create individual graphs
        4. Assign tissue type as graph-level label
        5. Create PyTorch Geometric Data objects

        Features are gene expression values that have been:
        - Log-transformed: log(counts + 1)
        - Filtered for genes present in all samples
        - Selected for high variability across samples
        """
        csv_data_path = Path(self.raw_dir) / RAW_FILE_NAME
        df = pd.read_csv(csv_data_path)

        logger.info(f"Loaded dataset: {len(df)} spots, {len(df.columns)} columns")

        # Identify gene expression feature columns
        gene_feature_cols = [col for col in df.columns if col not in METADATA_COLS]
        logger.info(f"Gene expression features: {len(gene_feature_cols)} genes")

        # Create mapping from codes to nominal values before encoding
        tissue_categorical = pd.Categorical(df[LABEL_COL])
        tissue_type_mapping = dict(enumerate(tissue_categorical.categories))

        # Encode tissue type labels as categorical codes
        df[LABEL_COL] = tissue_categorical.codes

        # Group by sample_id to create one graph per sample
        groups = list(df.groupby("sample_id"))
        data_list = []

        logger.info(
            f"Processing {len(groups)} samples from glioblastoma gene expression dataset"
        )

        for sample_id, sample_df in groups:
            # Verify that all spots in a sample have the same tissue type label
            labels = set(sample_df[LABEL_COL].values)
            assert len(labels) == 1, f"Sample {sample_id} has multiple labels: {labels}"

            # Graph-level label (tissue type)
            y = torch.tensor(list(labels)[0], dtype=torch.long)

            # Spatial coordinates for each spot
            pos = torch.tensor(sample_df[POS_COLS].values, dtype=torch.float)

            # Gene expression features
            feature_data = sample_df[gene_feature_cols].values
            x = torch.tensor(feature_data, dtype=torch.float)

            # Validate feature matrix
            assert not torch.isnan(
                x
            ).any(), f"NaN values found in features for sample {sample_id}"
            assert not torch.isinf(
                x
            ).any(), f"Infinite values found in features for sample {sample_id}"

            # Create PyTorch Geometric Data object
            data = Data(x=x, y=y, pos=pos)

            # Add sample metadata as attributes
            data.sample_id = sample_id
            data.patient_id = sample_df["patient_id"].iloc[0]
            data.num_spots = len(sample_df)
            data.num_genes = len(gene_feature_cols)

            data_list.append(data)

        logger.info(
            f"Created {len(data_list)} graphs from glioblastoma gene expression dataset"
        )

        # Log dataset statistics with both codes and nominal values
        tissue_type_counts = df.groupby(["sample_id", LABEL_COL]).size().reset_index()
        tissue_type_summary = tissue_type_counts.groupby(LABEL_COL).size()

        # Create distribution with both codes and nominal values
        distribution_with_names = {
            f"{code} ({tissue_type_mapping[code]})": count
            for code, count in tissue_type_summary.items()
        }
        logger.info(f"Tissue type distribution: {distribution_with_names}")

        # Log feature statistics
        logger.info(f"Average spots per sample: {len(df) / len(groups):.1f}")
        logger.info(f"Gene expression feature dimensions: {len(gene_feature_cols)}")

        # Log some example gene names
        example_genes = gene_feature_cols[:10]
        logger.info(f"Example genes: {example_genes}")

        return data_list
