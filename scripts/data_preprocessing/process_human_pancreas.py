#!/usr/bin/env python3
"""
Process GSE197317 human pancreas development spatial transcriptomics data.

This script processes 10x Visium spatial transcriptomics data from human pancreatic
development study, combining spatial coordinates, gene expression, and cell type
deconvolution data into a single CSV file for node classification tasks.

Usage:
    python scripts/data_preprocessing/process_human_pancreas.py <input_directory> [output_file]

Arguments:
    input_directory: Path to directory containing extracted GSE197317 data
    output_file: Optional output CSV file path (default: data/human-pancreas/raw/source.csv)

The input directory should contain:
- Extracted 10x Visium directories (e.g., 12PCW_S1/, 12PCW_S2/, etc.)
- Cell type deconvolution CSV files (e.g., GSM5914539_celltype_deconvolution_12PCW_section1.csv.gz)
"""

import argparse
import gzip
import os
from warnings import simplefilter

import pandas as pd
from scipy.io import mmread

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

NUM_TOP_GENES = 1_000


def load_10x_data(sample_dir):
    """Load 10x Visium data from a directory."""

    # Load spatial coordinates
    positions_file = os.path.join(sample_dir, "spatial", "tissue_positions_list.csv")
    if not os.path.exists(positions_file):
        raise FileNotFoundError(f"Positions file not found: {positions_file}")

    positions = pd.read_csv(
        positions_file,
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ],
    )

    # Load gene expression matrix
    matrix_file = os.path.join(sample_dir, "raw_feature_bc_matrix", "matrix.mtx.gz")
    features_file = os.path.join(sample_dir, "raw_feature_bc_matrix", "features.tsv.gz")
    barcodes_file = os.path.join(sample_dir, "raw_feature_bc_matrix", "barcodes.tsv.gz")

    if not all(os.path.exists(f) for f in [matrix_file, features_file, barcodes_file]):
        raise ValueError(f"Gene expression files not found in {sample_dir}.")

    # Load sparse matrix
    matrix = mmread(matrix_file).T.tocsc()  # transpose to get cells x genes

    # Load features and barcodes
    with gzip.open(features_file, "rt") as f:
        features = pd.read_csv(
            f, sep="\t", header=None, names=["gene_id", "gene_symbol", "feature_type"]
        )

    with gzip.open(barcodes_file, "rt") as f:
        barcodes = pd.read_csv(f, sep="\t", header=None, names=["barcode"])

    # Create gene expression dataframe
    gene_expr_df = pd.DataFrame.sparse.from_spmatrix(
        matrix, index=barcodes["barcode"], columns=features["gene_symbol"]
    )

    return positions, gene_expr_df


def load_deconvolution_data(deconv_file):
    """Load cell type deconvolution data."""
    with gzip.open(deconv_file, "rt") as f:
        deconv_df = pd.read_csv(f, index_col=0)
    return deconv_df.T  # transpose so barcodes are rows


def process_all_data(input_dir):
    """Process all samples and combine into a single dataset."""

    # Define sample mappings (directory_name, deconv_file, stage, section)
    samples = [
        (
            "12PCW_S2",
            "GSM5914539_celltype_deconvolution_12PCW_section1.csv.gz",
            "12PCW",
            "1",
        ),
        (
            "15PCW_S2",
            "GSM5914541_celltype_deconvolution_15PCW_section1.csv.gz",
            "15PCW",
            "1",
        ),
        (
            "15PCW_S1",
            "GSM5914542_celltype_deconvolution_15PCW_section2.csv.gz",
            "15PCW",
            "2",
        ),
        (
            "18PCW_S2",
            "GSM5914543_celltype_deconvolution_18PCW_section1.csv.gz",
            "18PCW",
            "1",
        ),
        (
            "18PCW_S1",
            "GSM5914544_celltype_deconvolution_18PCW_section2.csv.gz",
            "18PCW",
            "2",
        ),
        (
            "20PCW_S2",
            "GSM5914545_celltype_deconvolution_20PCW_section1.csv.gz",
            "20PCW",
            "1",
        ),
        (
            "20PCW_S1",
            "GSM5914546_celltype_deconvolution_20PCW_section2.csv.gz",
            "20PCW",
            "2",
        ),
    ]

    # Note: GSM5914540_celltype_deconvolution_12PCW_section2.csv.gz doesn't have corresponding spatial data

    all_data = []

    for sample_dir, deconv_file, stage, section in samples:
        print(f"Processing {sample_dir} ({stage}, section {section})...")

        sample_path = os.path.join(input_dir, sample_dir)
        deconv_path = os.path.join(input_dir, deconv_file)

        if not os.path.exists(sample_path):
            raise OSError(f"Directory not found: {sample_path}.")
            continue

        if not os.path.exists(deconv_path):
            raise OSError(f"Deconvolution file not found: {deconv_path}.")
            continue

        # Load spatial and expression data
        positions, gene_expr = load_10x_data(sample_path)

        # Load deconvolution data
        deconv = load_deconvolution_data(deconv_path)

        # Find common barcodes between spatial and deconvolution data
        common_barcodes = set(positions["barcode"]).intersection(set(deconv.index))

        if gene_expr is not None:
            common_barcodes = common_barcodes.intersection(set(gene_expr.index))

        print(f"  Common barcodes: {len(common_barcodes)}")

        if len(common_barcodes) == 0:
            raise ValueError(f"No common barcodes found.")

        # Convert to list for pandas indexing
        common_barcodes_list = list(common_barcodes)

        # Filter to common barcodes
        positions_filt = positions[
            positions["barcode"].isin(common_barcodes_list)
        ].set_index("barcode")
        deconv_filt = deconv.loc[common_barcodes_list]

        # Get dominant cell type for each spot
        cell_types = deconv_filt.idxmax(axis=1)

        # Combine data
        sample_data = positions_filt.copy()
        sample_data["stage"] = stage
        sample_data["section"] = section
        sample_data["sample_id"] = f"{stage}_S{section}"
        sample_data["cell_type"] = cell_types

        # Add gene expression if available
        if gene_expr is not None:
            gene_expr_filt = gene_expr.loc[common_barcodes_list]

            # Convert sparse to dense for variance calculation
            gene_expr_dense = gene_expr_filt.sparse.to_dense()

            # Calculate variance for each gene and keep top variable genes
            gene_vars = gene_expr_dense.var().sort_values(ascending=False)
            top_genes = gene_vars.head(NUM_TOP_GENES).index  # Keep top-k variable genes

            for gene in top_genes:
                sample_data[f"gene_{gene}"] = gene_expr_dense[gene].values

        # Add spatial coordinates as separate columns
        sample_data["x_coord"] = sample_data["pxl_col_in_fullres"]
        sample_data["y_coord"] = sample_data["pxl_row_in_fullres"]

        # Add cell type proportions as features
        for cell_type in deconv_filt.columns:
            sample_data[f"prop_{cell_type}"] = deconv_filt[cell_type].values

        all_data.append(sample_data.reset_index())

    # Combine all samples
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Cell types distribution:")
    print(final_df["cell_type"].value_counts())
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_directory", help="Path to directory containing extracted GSE197317 data"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/human-pancreas/raw/source.csv",
        help="Output CSV file path (default: data/human-pancreas/raw/source.csv)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_directory)
    output_file = args.output

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Processing data from: {input_dir}")
    print(f"Output file: {output_file}")

    # Process data
    df = process_all_data(input_dir)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    print(f"Total cells: {len(df)}")
    print(f"Features: {len(df.columns)}")


if __name__ == "__main__":
    main()
