import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from logzero import logger
from torch_geometric.data import Data

from .base import AbstractDataset


def _mem_gb():
    """Return current process RSS in GB."""
    import resource

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    if os.uname().sysname == "Darwin":
        return rss_kb / 1e9
    return rss_kb / 1e6


ID_COL = "barcodes"
GROUP_COLS = ["Source-Sample-ID", "Sample_title"]
LABEL_COL = "Disease-Status"
POS_COLS = ["xcoord", "ycoord"]
COLS_TO_DROP = ["cell_type", "Tissue", "Age"]
RAW_FILE_NAME = "GSE190094.parquet"
N_TOP_GENES = 1000
HVG_SAMPLE_SIZE = 500_000  # rows sampled for variance estimation (~25% of 2M cells)


def select_hvg(gene_df: pd.DataFrame, n_top: int) -> list[str]:
    """Select top n_top highly variable genes by variance across all cells."""
    variances = gene_df.var()
    return variances.nlargest(n_top).index.tolist()


def _select_hvg_from_sample(
    data_path: Path, gene_cols: list[str], n_top: int, sample_size: int
) -> tuple[list[str], set[str]]:
    """Sample rows to estimate gene variances; return (top_gene_cols, nan_cols)."""
    pf = pq.ParquetFile(data_path)
    batch = next(pf.iter_batches(batch_size=sample_size, columns=gene_cols))
    sample_df = batch.to_pandas()

    nan_cols = {c for c in sample_df.columns if sample_df[c].isna().any()}
    if nan_cols:
        logger.info(f"Dropping NaN columns (detected from sample): {nan_cols}")
        sample_df = sample_df.drop(columns=nan_cols)

    return select_hvg(sample_df, n_top), nan_cols


class MouseKidneyDataset(AbstractDataset):
    """
    Example:
    >> from stgym.data_loader.mouse_kidney import MouseKidneyDataset
    >> ds = MouseKidneyDataset(root="../data/mouse-kidney")
    """

    @property
    def raw_file_names(self):
        return [RAW_FILE_NAME]

    def process_data(self):
        # Two-pass loading strategy to avoid OOM:
        # Loading all 2M cells × 2,000 genes as a dense float32 DataFrame requires ~33 GB
        # RSS (parquet decompresses ~100x in memory), which exceeds available RAM under
        # concurrent workloads. Pass 1 uses a small row sample (~800 MB) to identify the
        # top-N_TOP_GENES by variance; pass 2 re-reads the full dataset with only those
        # columns via parquet column pruning, reducing peak memory to ~8 GB.
        data_path = Path(self.raw_dir) / RAW_FILE_NAME
        all_cols = pq.read_schema(data_path).names
        cols_to_skip = set(COLS_TO_DROP + [ID_COL])
        non_feature_cols = set(GROUP_COLS + POS_COLS + [LABEL_COL])
        all_gene_cols = [
            c for c in all_cols if c not in cols_to_skip | non_feature_cols
        ]

        # Pass 1: sample rows to select HVGs and detect NaN columns
        logger.info(
            f"[mem {_mem_gb():.1f} GB] pass 1: sampling {HVG_SAMPLE_SIZE} rows for HVG selection..."
        )
        feature_cols, nan_cols = _select_hvg_from_sample(
            data_path, all_gene_cols, N_TOP_GENES, HVG_SAMPLE_SIZE
        )
        logger.info(f"[mem {_mem_gb():.1f} GB] selected {len(feature_cols)} HVGs")

        # Pass 2: load all rows but only selected columns
        cols_to_read = [
            c
            for c in all_cols
            if c not in cols_to_skip | nan_cols
            and (c in non_feature_cols or c in feature_cols)
        ]
        logger.info(
            f"[mem {_mem_gb():.1f} GB] pass 2: reading {len(cols_to_read)} columns..."
        )
        df = pd.read_parquet(data_path, columns=cols_to_read)
        logger.info(
            f"[mem {_mem_gb():.1f} GB] loaded DataFrame: "
            f"{df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.1f} GB"
        )

        # Encode labels in-place
        df[LABEL_COL] = pd.Categorical(df[LABEL_COL]).codes

        logger.info(f"[mem {_mem_gb():.1f} GB] before building graphs")

        data_list = []
        for i, (_, sample_df) in enumerate(df.groupby(GROUP_COLS)):
            labels = sample_df[LABEL_COL].unique()
            assert len(labels) == 1, len(labels)
            y = torch.tensor(labels[0], dtype=torch.long)
            pos = torch.tensor(sample_df[POS_COLS].values, dtype=torch.float)
            x = torch.tensor(sample_df[feature_cols].values, dtype=torch.float)
            data_list.append(Data(x=x, y=y, pos=pos))
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"[mem {_mem_gb():.1f} GB] processed graph {i + 1}")

        logger.info(f"[mem {_mem_gb():.1f} GB] all {len(data_list)} graphs done")
        return data_list
