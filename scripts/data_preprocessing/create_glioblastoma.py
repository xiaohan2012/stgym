#!/usr/bin/env python3
"""
Fast gene expression-based data consolidation script for glioblastoma spatial transcriptomics dataset.

This optimized version processes 10X Visium data more efficiently by using filtered matrices
and smart gene selection to avoid computing variance on all 33K+ genes.

Usage:
    python scripts/data_preprocessing/create_glioblastoma.py \
        --input-dir ./10XVisium_2 \
        --output-dir ./data/glioblastoma/raw
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from logzero import logger


def extract_patient_id(sample_name):
    """Extract patient ID from sample name (e.g., #UKF304_T_ST -> UKF304)"""
    match = re.search(r"#?(UKF\d+)", sample_name)
    return match.group(1) if match else None


def extract_tissue_type(sample_name):
    """Extract tissue type from sample name (T=tumor, C=cortex, etc.)"""
    if "_T_" in sample_name or sample_name.endswith("_T_ST"):
        return "tumor"
    elif "_C_" in sample_name or sample_name.endswith("_C_ST"):
        return "cortex"
    elif "_TC_" in sample_name:
        return "tumor_core"
    elif "_TI_" in sample_name:
        return "tumor_infiltration"
    else:
        return "unknown"


def load_10x_h5_fast(h5_file):
    """Load 10X HDF5 format gene expression matrix (optimized version)."""
    logger.info(f"  Loading HDF5 matrix: {h5_file}")

    with h5py.File(h5_file, "r") as f:
        matrix = f["matrix"]

        # Load sparse matrix components
        data = matrix["data"][...]
        indices = matrix["indices"][...]
        indptr = matrix["indptr"][...]
        shape = matrix["shape"][...]

        # Create sparse matrix (genes x barcodes)
        sparse_matrix = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)

        # Load barcodes (spots)
        barcodes = [bc.decode("utf-8") for bc in matrix["barcodes"][...]]

        # Load gene features
        features = matrix["features"]
        gene_ids = [gid.decode("utf-8") for gid in features["id"][...]]
        gene_names = [gname.decode("utf-8") for gname in features["name"][...]]

        return sparse_matrix, barcodes, gene_ids, gene_names


def load_10x_mtx(matrix_dir):
    """Load 10X Matrix Market format gene expression matrix."""
    logger.info(f"  Loading MTX matrix: {matrix_dir}")

    # Load matrix
    matrix_file = os.path.join(matrix_dir, "matrix.mtx.gz")
    if not os.path.exists(matrix_file):
        matrix_file = os.path.join(matrix_dir, "matrix.mtx")

    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Matrix file not found in {matrix_dir}")

    sparse_matrix = scipy.io.mmread(matrix_file).tocsc()

    # Load barcodes
    barcode_file = os.path.join(matrix_dir, "barcodes.tsv.gz")
    if not os.path.exists(barcode_file):
        barcode_file = os.path.join(matrix_dir, "barcodes.tsv")

    barcodes = pd.read_csv(barcode_file, header=None, sep="\t")[0].tolist()

    # Load features
    features_file = os.path.join(matrix_dir, "features.tsv.gz")
    if not os.path.exists(features_file):
        features_file = os.path.join(matrix_dir, "features.tsv")

    features_df = pd.read_csv(features_file, header=None, sep="\t")
    gene_ids = features_df[0].tolist()
    gene_names = features_df[1].tolist()

    return sparse_matrix, barcodes, gene_ids, gene_names


def load_tissue_positions(sample_dir):
    """Load tissue position information."""
    tissue_pos_file = os.path.join(sample_dir, "outs/spatial/tissue_positions_list.csv")

    if not os.path.exists(tissue_pos_file):
        logger.warning(f"    No tissue positions file found in {sample_dir}")
        return None

    # Read tissue positions
    pos_df = pd.read_csv(
        tissue_pos_file,
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "x_coord", "y_coord"],
    )

    # Only keep spots that are in tissue
    pos_df = pos_df[pos_df["in_tissue"] == 1].copy()

    return pos_df


def load_sample_gene_expression_fast(sample_dir):
    """Load gene expression data for a single sample (optimized version)."""
    sample_name = os.path.basename(sample_dir)
    logger.info(f"Processing sample: {sample_name}")

    # Load tissue positions first
    pos_df = load_tissue_positions(sample_dir)
    if pos_df is None or len(pos_df) == 0:
        logger.warning(f"  Skipping {sample_name} - no tissue positions")
        return None

    # Use filtered matrix for faster processing (smaller, already QC'd)
    filtered_h5_file = os.path.join(sample_dir, "outs/filtered_feature_bc_matrix.h5")
    filtered_mtx_dir = os.path.join(sample_dir, "outs/filtered_feature_bc_matrix")

    sparse_matrix = None
    barcodes = None
    gene_ids = None
    gene_names = None
    matrix_type = None

    try:
        # Try filtered data (much smaller and faster)
        if os.path.exists(filtered_h5_file):
            sparse_matrix, barcodes, gene_ids, gene_names = load_10x_h5_fast(
                filtered_h5_file
            )
            matrix_type = "filtered_h5"
        elif os.path.exists(filtered_mtx_dir):
            sparse_matrix, barcodes, gene_ids, gene_names = load_10x_mtx(
                filtered_mtx_dir
            )
            matrix_type = "filtered_mtx"
        else:
            logger.error(
                f"  No filtered gene expression matrix found for {sample_name}"
            )
            return None

        logger.info(
            f"  Loaded {matrix_type}: {sparse_matrix.shape[0]} genes × {sparse_matrix.shape[1]} barcodes"
        )

        # Match barcodes with tissue positions
        tissue_barcodes = set(pos_df["barcode"])
        expression_barcodes = set(barcodes)
        matching_barcodes = tissue_barcodes.intersection(expression_barcodes)

        if len(matching_barcodes) == 0:
            logger.error(
                f"  No matching barcodes between tissue positions and expression data for {sample_name}"
            )
            return None

        logger.info(
            f"  Found {len(matching_barcodes)} spots with both position and expression data"
        )

        # Extract expression data for tissue spots only
        barcode_to_idx = {bc: idx for idx, bc in enumerate(barcodes)}
        tissue_indices = [
            barcode_to_idx[bc] for bc in matching_barcodes if bc in barcode_to_idx
        ]

        if not tissue_indices:
            logger.error(f"  No valid tissue indices found for {sample_name}")
            return None

        # Get submatrix for tissue spots (genes x spots)
        tissue_matrix = sparse_matrix[:, tissue_indices]
        tissue_barcodes_list = [barcodes[i] for i in tissue_indices]

        # Filter tissue positions to matching barcodes and sort to match expression matrix
        pos_df_filtered = pos_df[pos_df["barcode"].isin(tissue_barcodes_list)].copy()
        pos_df_filtered = (
            pos_df_filtered.set_index("barcode")
            .reindex(tissue_barcodes_list)
            .reset_index()
        )

        # Add sample metadata
        patient_id = extract_patient_id(sample_name)
        tissue_type = extract_tissue_type(sample_name)

        pos_df_filtered["sample_id"] = sample_name
        pos_df_filtered["patient_id"] = patient_id
        pos_df_filtered["tissue_type"] = tissue_type

        return {
            "sample_id": sample_name,
            "patient_id": patient_id,
            "tissue_type": tissue_type,
            "position_data": pos_df_filtered,
            "expression_matrix": tissue_matrix,
            "gene_ids": gene_ids,
            "gene_names": gene_names,
            "barcodes": tissue_barcodes_list,
            "matrix_type": matrix_type,
        }

    except Exception as e:
        logger.error(f"  Error loading gene expression for {sample_name}: {e}")
        return None


def smart_gene_filtering(
    well_expressed_genes,
    gene_total_expression,
    gene_id_to_name,
    sample_data_list,
    max_genes=5000,
):
    """Smart gene filtering to avoid random sampling."""

    if len(well_expressed_genes) <= max_genes:
        logger.info(
            f"No filtering needed: {len(well_expressed_genes)} genes <= {max_genes}"
        )
        return well_expressed_genes

    logger.info(
        f"Smart filtering: reducing {len(well_expressed_genes)} genes to {max_genes}"
    )

    # Strategy 1: Progressive expression threshold increase
    logger.info("Strategy 1: Increasing expression thresholds...")
    for threshold in [200, 500, 1000, 2000, 5000, 10000]:
        filtered = [
            g for g in well_expressed_genes if gene_total_expression[g] >= threshold
        ]
        logger.info(f"  Expression threshold {threshold}: {len(filtered)} genes")
        if len(filtered) <= max_genes:
            if len(filtered) >= max_genes // 2:  # Don't filter too aggressively
                logger.info(
                    f"Selected {len(filtered)} genes using expression threshold {threshold}"
                )
                return filtered

    # Strategy 2: Prioritize known important gene categories for glioblastoma
    logger.info("Strategy 2: Prioritizing biologically relevant genes...")
    priority_patterns = [
        # Glioblastoma-relevant genes
        "TP53",
        "EGFR",
        "PTEN",
        "IDH",
        "MGMT",
        "ATRX",
        "CIC",
        "FUBP1",
        # Neural/glial markers
        "OLIG",
        "SOX",
        "GFAP",
        "NESTIN",
        "VIM",
        "S100",
        # Immune markers
        "CD",
        "IL",
        "TNF",
        "IFNG",
        "FOXP3",
        # Cell cycle/proliferation
        "MKI67",
        "PCNA",
        "TOP2A",
        "CCND",
        "CCNE",
        "CDK",
        # Apoptosis/survival
        "BCL",
        "BAX",
        "CASP",
        "BIRC",
        # Signaling pathways
        "AKT",
        "MTOR",
        "PI3K",
        "RB1",
        "MDM",
        "MYC",
        # Hypoxia/metabolism
        "HIF",
        "VEGF",
        "PDGF",
    ]

    priority_genes = []
    remaining_genes = []

    for gene_id in well_expressed_genes:
        gene_name = gene_id_to_name.get(gene_id, gene_id)
        is_priority = any(pattern in gene_name.upper() for pattern in priority_patterns)

        if is_priority:
            priority_genes.append(gene_id)
        else:
            remaining_genes.append(gene_id)

    logger.info(f"  Found {len(priority_genes)} priority genes")

    if len(priority_genes) >= max_genes:
        # Too many priority genes, sort by expression
        priority_genes.sort(key=lambda g: gene_total_expression[g], reverse=True)
        logger.info(f"Selected top {max_genes} priority genes by expression")
        return priority_genes[:max_genes]

    # Strategy 3: Single-sample variance estimation for remaining slots
    remaining_slots = max_genes - len(priority_genes)
    if remaining_slots > 0 and remaining_genes:
        logger.info(
            f"Strategy 3: Single-sample variance for {remaining_slots} additional genes..."
        )

        # Use first sample for quick variance estimation
        first_sample = sample_data_list[0]
        gene_variances = {}

        for gene_id in remaining_genes:
            if gene_id in first_sample["gene_ids"]:
                gene_idx = first_sample["gene_ids"].index(gene_id)
                expr = (
                    first_sample["expression_matrix"][gene_idx, :].toarray().flatten()
                )
                # Quick variance calculation on log-transformed data
                log_expr = np.log1p(expr)
                gene_variances[gene_id] = (
                    np.var(log_expr) * gene_total_expression[gene_id]
                )  # Weight by expression
            else:
                gene_variances[gene_id] = 0

        # Select top variable genes
        top_variable = sorted(
            gene_variances.keys(), key=lambda x: gene_variances[x], reverse=True
        )[:remaining_slots]

        final_genes = priority_genes + top_variable
        logger.info(
            f"Final selection: {len(priority_genes)} priority + {len(top_variable)} variable = {len(final_genes)} genes"
        )
        return final_genes

    logger.info(f"Final selection: {len(priority_genes)} priority genes only")
    return priority_genes


def create_universal_gene_set_fast(
    sample_data_list, min_expression=100, top_genes=1000
):
    """Create universal gene set across all samples (optimized version with smart filtering)."""
    logger.info("Creating universal gene set (fast version with smart filtering)...")

    if not sample_data_list:
        raise ValueError("No valid samples found")

    # Step 1: Find genes present in ALL samples and compute total expression efficiently
    logger.info("Step 1: Finding universal genes and computing expression totals...")

    # Get gene sets from each sample
    sample_gene_sets = []
    gene_total_expression = Counter()
    gene_id_to_name = {}

    for i, sample_data in enumerate(sample_data_list):
        gene_ids = sample_data["gene_ids"]
        gene_names = sample_data["gene_names"]
        expression_matrix = sample_data["expression_matrix"]

        sample_gene_set = set(gene_ids)
        sample_gene_sets.append(sample_gene_set)

        # Update gene name mapping
        for gid, gname in zip(gene_ids, gene_names):
            gene_id_to_name[gid] = gname

        # Compute total expression per gene for this sample
        gene_sums = np.array(expression_matrix.sum(axis=1)).flatten()
        for gene_idx, gene_id in enumerate(gene_ids):
            gene_total_expression[gene_id] += gene_sums[gene_idx]

        logger.info(f"  Processed sample {i+1}/{len(sample_data_list)}")

    # Find intersection of all gene sets (genes present in ALL samples)
    universal_gene_set = sample_gene_sets[0]
    for gene_set in sample_gene_sets[1:]:
        universal_gene_set = universal_gene_set.intersection(gene_set)

    universal_genes = list(universal_gene_set)
    logger.info(
        f"Genes present in all {len(sample_data_list)} samples: {len(universal_genes)}"
    )

    # Step 2: Filter by minimum expression
    logger.info(f"Step 2: Filtering genes by minimum expression ({min_expression})...")
    well_expressed_genes = [
        gid for gid in universal_genes if gene_total_expression[gid] >= min_expression
    ]
    logger.info(
        f"Genes with >= {min_expression} total expression: {len(well_expressed_genes)}"
    )

    if len(well_expressed_genes) == 0:
        logger.warning("No genes pass expression filter! Using all universal genes...")
        well_expressed_genes = universal_genes

    # Step 3: Smart gene selection (NO random sampling)
    logger.info("Step 3: Smart gene selection for feature matrix...")
    selected_genes = smart_gene_filtering(
        well_expressed_genes,
        gene_total_expression,
        gene_id_to_name,
        sample_data_list,
        max_genes=top_genes,
    )

    logger.info(f"Final gene selection: {len(selected_genes)} genes")

    return selected_genes, gene_id_to_name


def consolidate_samples_fast(sample_data_list, selected_genes, gene_id_to_name):
    """Consolidate all samples into a single DataFrame (optimized version)."""
    logger.info("Consolidating samples into single dataset...")

    all_sample_dfs = []

    for i, sample_data in enumerate(sample_data_list):
        logger.info(
            f"  Processing sample {i+1}/{len(sample_data_list)}: {sample_data['sample_id']}"
        )

        # Get position data
        pos_df = sample_data["position_data"].copy()

        # Get expression data for selected genes
        gene_ids = sample_data["gene_ids"]
        expression_matrix = sample_data["expression_matrix"]

        # Create expression DataFrame efficiently
        gene_expression_data = {}

        for j, gene_id in enumerate(selected_genes):
            if gene_id in gene_ids:
                gene_idx = gene_ids.index(gene_id)
                gene_expression = expression_matrix[gene_idx, :].toarray().flatten()
                # Apply log transformation
                log_expression = np.log1p(gene_expression)
            else:
                # Gene not present in this sample, fill with zeros
                log_expression = np.zeros(len(pos_df))

            # Use gene symbol as column name, fallback to gene ID if symbol unavailable
            gene_name = gene_id_to_name.get(gene_id, gene_id)
            col_name = gene_name if gene_name and gene_name != gene_id else gene_id
            gene_expression_data[col_name] = log_expression

            if (j + 1) % 100 == 0:
                logger.info(f"    Processed {j+1}/{len(selected_genes)} genes")

        # Create expression DataFrame
        expr_df = pd.DataFrame(gene_expression_data, index=range(len(pos_df)))

        # Combine with position data
        sample_df = pd.concat([pos_df.reset_index(drop=True), expr_df], axis=1)
        all_sample_dfs.append(sample_df)

    # Concatenate all samples
    consolidated_df = pd.concat(all_sample_dfs, ignore_index=True)
    logger.info(
        f"Consolidated dataset: {len(consolidated_df)} spots across {len(sample_data_list)} samples"
    )

    return consolidated_df


def create_consolidated_gene_expression_dataset_fast(
    input_dir, output_dir, min_expression=100, top_genes=1000
):
    """Create consolidated gene expression dataset from 10X Visium data (optimized version)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all sample directories
    sample_dirs = [d for d in glob.glob(f"{input_path}/#UKF*") if os.path.isdir(d)]
    logger.info(f"Found {len(sample_dirs)} sample directories")

    if len(sample_dirs) == 0:
        raise ValueError(f"No sample directories found in {input_path}")

    # Load all samples (using filtered matrices for speed)
    logger.info(
        "Loading gene expression data from all samples (using filtered matrices)..."
    )
    sample_data_list = []

    for sample_dir in sorted(sample_dirs):
        sample_data = load_sample_gene_expression_fast(sample_dir)
        if sample_data is not None:
            sample_data_list.append(sample_data)

    if not sample_data_list:
        raise ValueError("No valid samples with gene expression data found!")

    logger.info(f"Successfully loaded {len(sample_data_list)} samples")

    # Create universal gene set (optimized with smart filtering)
    selected_genes, gene_id_to_name = create_universal_gene_set_fast(
        sample_data_list, min_expression=min_expression, top_genes=top_genes
    )

    # Consolidate all samples (optimized)
    consolidated_df = consolidate_samples_fast(
        sample_data_list, selected_genes, gene_id_to_name
    )

    # Generate summary statistics
    logger.info("Dataset statistics:")
    logger.info(f"  Total spots: {len(consolidated_df):,}")
    logger.info(f"  Unique samples: {consolidated_df['sample_id'].nunique()}")
    logger.info(f"  Unique patients: {consolidated_df['patient_id'].nunique()}")
    logger.info(f"  Gene features: {len(selected_genes)}")
    logger.info(f"  Tissue types: {sorted(consolidated_df['tissue_type'].unique())}")

    # Validate data consistency
    logger.info("Validating data consistency...")
    sample_tissue_counts = consolidated_df.groupby("sample_id")["tissue_type"].nunique()
    inconsistent_samples = sample_tissue_counts[sample_tissue_counts > 1]

    if len(inconsistent_samples) > 0:
        logger.error(
            f"Found {len(inconsistent_samples)} samples with multiple tissue types!"
        )
        for sample_id in inconsistent_samples.index:
            types = consolidated_df[consolidated_df["sample_id"] == sample_id][
                "tissue_type"
            ].unique()
            logger.error(f"  {sample_id}: {types}")
        raise ValueError("Data consistency check failed")

    # Save consolidated dataset
    output_file = output_path / "source.csv"
    consolidated_df.to_csv(output_file, index=False)
    logger.info(f"Saved consolidated dataset to: {output_file}")

    # Save gene information
    gene_info_df = pd.DataFrame(
        {
            "gene_id": selected_genes,
            "gene_name": [gene_id_to_name[gid] for gid in selected_genes],
        }
    )
    gene_info_file = output_path / "gene_info.csv"
    gene_info_df.to_csv(gene_info_file, index=False)
    logger.info(f"Saved gene information to: {gene_info_file}")

    # Save summary statistics
    summary_file = output_path / "dataset_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Glioblastoma Gene Expression Dataset Summary (Fast Processing)\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            f"Processing date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Data source: Filtered 10X Visium gene expression matrices\n\n")
        f.write(f"Total spots: {len(consolidated_df):,}\n")
        f.write(f"Total samples: {consolidated_df['sample_id'].nunique()}\n")
        f.write(f"Unique patients: {consolidated_df['patient_id'].nunique()}\n")
        f.write(f"Gene features: {len(selected_genes)}\n")
        f.write(f"Tissue types: {len(consolidated_df['tissue_type'].unique())}\n\n")

        tissue_counts = (
            consolidated_df.groupby("tissue_type")
            .agg({"sample_id": "nunique", "barcode": "count"})
            .rename(columns={"sample_id": "samples", "barcode": "spots"})
        )

        f.write("Tissue type distribution:\n")
        for tissue_type, row in tissue_counts.iterrows():
            f.write(
                f"  {tissue_type}: {row['samples']} samples, {row['spots']:,} spots\n"
            )

        f.write(f"\nFeature columns (first 20 genes):\n")
        gene_cols = [
            col
            for col in consolidated_df.columns
            if col
            not in [
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
        ]
        for i, col in enumerate(gene_cols[:20]):
            f.write(f"  {col}\n")
        if len(gene_cols) > 20:
            f.write(f"  ... and {len(gene_cols) - 20} more genes\n")

    logger.info(f"Saved summary to: {summary_file}")

    return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create consolidated glioblastoma gene expression dataset (optimized version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/data_preprocessing/create_glioblastoma.py \\
      --input-dir "/Users/misc/Downloads/doi_10_5061_dryad_h70rxwdmj__v20250306/10XVisium 2" \\
      --output-dir ./data/glioblastoma/raw \\
      --top-genes 1000
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to extracted 10XVisium_2 directory from Dryad",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/glioblastoma/raw",
        help="Output directory for processed dataset",
    )

    parser.add_argument(
        "--min-expression",
        type=int,
        default=100,
        help="Minimum total expression across all samples for gene inclusion",
    )

    parser.add_argument(
        "--top-genes",
        type=int,
        default=1000,
        help="Number of top variable genes to select",
    )

    args = parser.parse_args()

    try:
        output_file = create_consolidated_gene_expression_dataset_fast(
            args.input_dir,
            args.output_dir,
            min_expression=args.min_expression,
            top_genes=args.top_genes,
        )
        logger.info(f"Fast dataset creation completed successfully!")
        logger.info(f"Output file: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
