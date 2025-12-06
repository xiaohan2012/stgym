#!/usr/bin/env python3
"""
Preprocessing script for gastric-bladder-cancer dataset from GSE246011.

This script merges metadata and count data from multiple samples into a single CSV
suitable for graph classification tasks.
"""

import gzip
from pathlib import Path

import pandas as pd


def uncompress_gz_files(data_dir):
    """Uncompress .gz files if needed."""
    data_dir = Path(data_dir)
    for gz_file in data_dir.glob("*.gz"):
        if gz_file.name.endswith(".csv.gz"):
            output_file = gz_file.with_suffix("")
            if not output_file.exists():
                print(f"Uncompressing {gz_file}")
                with gzip.open(gz_file, "rt") as f_in:
                    with open(output_file, "w") as f_out:
                        f_out.write(f_in.read())


def load_sample_data(sample_prefix, data_dir):
    """Load metadata and count data for a single sample."""
    data_dir = Path(data_dir)

    # Find metadata file
    metadata_file = None
    for f in data_dir.glob(f"{sample_prefix}*Metadata*.csv"):
        if not f.name.endswith(".gz"):
            metadata_file = f
            break

    if not metadata_file:
        raise FileNotFoundError(f"No metadata file found for {sample_prefix}")

    # Find count file
    count_file = None
    for f in data_dir.glob(f"{sample_prefix}*count*.csv"):
        if not f.name.endswith(".gz"):
            count_file = f
            break

    if not count_file:
        raise FileNotFoundError(f"No count file found for {sample_prefix}")

    print(f"Loading {sample_prefix}: {metadata_file.name} + {count_file.name}")

    # Load metadata
    metadata = pd.read_csv(metadata_file, index_col=0)

    # Load count data
    counts = pd.read_csv(count_file, index_col=0)

    # Ensure indices match
    common_indices = metadata.index.intersection(counts.index)
    print(
        f"  Common barcodes: {len(common_indices)} (metadata: {len(metadata)}, counts: {len(counts)})"
    )

    metadata = metadata.loc[common_indices]
    counts = counts.loc[common_indices]

    return metadata, counts


def create_merged_dataset():
    """Create merged dataset from all samples."""
    data_dir = Path("data/gastric-bladder-cancer/raw/GSE246011_extracted")

    # Uncompress files if needed
    print("Uncompressing files...")
    uncompress_gz_files(data_dir)

    # Sample information: (prefix, sample_id, cancer_type)
    samples = [
        ("GSM7853983_STAD-G1", "STAD-G1", "STAD"),
        ("GSM7853984_STAD-G2", "STAD-G2", "STAD"),
        ("GSM7853985_STAD-G3", "STAD-G3", "STAD"),
        ("GSM7853986_STAD-G4", "STAD-G4", "STAD"),
        ("GSM7853987_BLCA-B1", "BLCA-B1", "BLCA"),
        ("GSM7853988_BLCA-B2", "BLCA-B2", "BLCA"),
    ]

    all_data = []

    for prefix, sample_id, cancer_type in samples:
        try:
            metadata, counts = load_sample_data(prefix, data_dir)

            # Add sample and cancer type information
            sample_data = metadata.copy()
            sample_data["sample_id"] = sample_id
            sample_data["cancer_type"] = cancer_type

            # Add gene expression data
            sample_data = pd.concat([sample_data, counts], axis=1)

            all_data.append(sample_data)
            print(f"  Added {len(sample_data)} cells from {sample_id}")

        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    # Merge all samples
    if not all_data:
        raise RuntimeError("No data loaded successfully")

    merged_df = pd.concat(all_data, axis=0, ignore_index=False)

    # Reset index to use cell barcodes as a column
    merged_df = merged_df.reset_index()
    merged_df = merged_df.rename(columns={"index": "barcode"})

    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"Samples: {merged_df['sample_id'].value_counts().to_dict()}")
    print(f"Cancer types: {merged_df['cancer_type'].value_counts().to_dict()}")

    # Save merged dataset
    output_file = "data/gastric-bladder-cancer/raw/source.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved merged dataset to: {output_file}")

    return merged_df


if __name__ == "__main__":
    create_merged_dataset()
