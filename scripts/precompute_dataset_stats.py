#!/usr/bin/env python3
"""Precompute dataset statistics for fast memory estimation.

This script precomputes dataset statistics (node counts, edge counts, etc.)
for different graph construction configurations and saves them as JSON cache files.
This eliminates the need to iterate through entire datasets during memory estimation,
which can be slow on CUDA machines when forced to use CPU-only mode.

Usage:
    python scripts/memory/precompute_dataset_stats.py --dataset brca-test --graph-const knn --knn-k 10
    python scripts/memory/precompute_dataset_stats.py --dataset mouse-spleen --graph-const radius --radius-ratio 0.1
"""

import argparse
import sys
from pathlib import Path

from stgym.cache import (
    DatasetStatistics,
    generate_cache_key,
    get_cache_file_path,
    load_statistics_from_cache,
    save_statistics_to_cache,
)
from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader.ds_info import get_all_ds_names, get_info
from stgym.mem_utils import compute_dataset_statistics_using_config


def compute_dataset_statistics(
    dataset_name: str, graph_const: str, knn_k: int = None, radius_ratio: float = None
) -> DatasetStatistics:
    """Compute dataset statistics for given configuration."""
    print(f"Computing statistics for {dataset_name} with {graph_const} construction...")

    # Create minimal task and dataloader configs
    task_cfg = TaskConfig(
        dataset_name=dataset_name,
        type=get_info(dataset_name)["task_type"],
        num_classes=get_info(dataset_name)["num_classes"],
    )

    dl_cfg = DataLoaderConfig(
        batch_size=32,  # Not used for statistics computation
        device="cpu",
        graph_const=graph_const,
        knn_k=knn_k,
        radius_ratio=radius_ratio,
        num_workers=0,
        split=DataLoaderConfig.DataSplitConfig(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        ),
    )

    # Use the function from mem_utils to compute statistics
    return compute_dataset_statistics_using_config(task_cfg, dl_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute dataset statistics for memory estimation"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g., brca-test, mouse-spleen)"
    )
    parser.add_argument(
        "--graph-const",
        required=True,
        choices=["knn", "radius"],
        help="Graph construction method",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        help="Number of nearest neighbors (for knn graph construction)",
    )
    parser.add_argument(
        "--radius-ratio",
        type=float,
        help="Radius ratio (for radius graph construction)",
    )
    parser.add_argument(
        "--cache-dir", default="./data/dataset_stats_cache", help="Cache directory path"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recomputation even if cache exists"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate cached statistics by recomputing",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.graph_const == "knn" and args.knn_k is None:
        parser.error("--knn-k is required when using knn graph construction")
    if args.graph_const == "radius" and args.radius_ratio is None:
        parser.error("--radius-ratio is required when using radius graph construction")

    # Check if dataset exists
    available_datasets = get_all_ds_names()
    if args.dataset not in available_datasets:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {', '.join(available_datasets)}")
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_key = generate_cache_key(
        args.dataset, args.graph_const, args.knn_k, args.radius_ratio
    )
    cache_file = get_cache_file_path(cache_key, cache_dir)

    print(f"Dataset: {args.dataset}")
    print(f"Graph construction: {args.graph_const}")
    if args.graph_const == "knn":
        print(f"KNN k: {args.knn_k}")
    else:
        print(f"Radius ratio: {args.radius_ratio}")
    print(f"Cache key: {cache_key}")
    print(f"Cache file: {cache_file}")
    print("-" * 60)

    # Check if cache exists and handle accordingly
    if cache_file.exists() and not args.force and not args.validate:
        print(f"✅ Cache already exists: {cache_file}")
        print("Use --force to recompute or --validate to verify existing cache")

        # Load and display cached statistics
        stats = load_statistics_from_cache(cache_key, cache_dir)
        print("\nCached statistics:")
        print(f"  Features: {stats.num_features}")
        print(f"  Graphs: {stats.num_graphs}")
        print(f"  Avg nodes: {stats.avg_nodes:.1f}, Max nodes: {stats.max_nodes}")
        print(f"  Avg edges: {stats.avg_edges:.1f}, Max edges: {stats.max_edges}")

    # Compute statistics
    try:
        stats = compute_dataset_statistics(
            args.dataset, args.graph_const, args.knn_k, args.radius_ratio
        )

        if args.validate and cache_file.exists():
            # Compare with cached statistics
            try:
                cached_stats = load_statistics_from_cache(cache_key, cache_dir)
                print(f"\n📊 VALIDATION RESULTS:")
                print(
                    f"  Features: cached={cached_stats.num_features}, computed={stats.num_features}"
                )
                print(
                    f"  Graphs: cached={cached_stats.num_graphs}, computed={stats.num_graphs}"
                )
                print(
                    f"  Avg nodes: cached={cached_stats.avg_nodes:.1f}, computed={stats.avg_nodes:.1f}"
                )
                print(
                    f"  Avg edges: cached={cached_stats.avg_edges:.1f}, computed={stats.avg_edges:.1f}"
                )
                print(
                    f"  Max nodes: cached={cached_stats.max_nodes}, computed={stats.max_nodes}"
                )
                print(
                    f"  Max edges: cached={cached_stats.max_edges}, computed={stats.max_edges}"
                )

                # Check for significant differences
                tolerance = 1e-6
                issues = []
                if abs(cached_stats.avg_nodes - stats.avg_nodes) > tolerance:
                    issues.append("avg_nodes")
                if abs(cached_stats.avg_edges - stats.avg_edges) > tolerance:
                    issues.append("avg_edges")
                if cached_stats.max_nodes != stats.max_nodes:
                    issues.append("max_nodes")
                if cached_stats.max_edges != stats.max_edges:
                    issues.append("max_edges")

                if issues:
                    print(
                        f"⚠️  Validation FAILED: Differences found in {', '.join(issues)}"
                    )
                else:
                    print(
                        f"✅ Validation PASSED: Cached statistics match computed values"
                    )

            except Exception as e:
                print(f"Error validating cache: {e}")
        else:
            # Save new statistics
            save_statistics_to_cache(stats, cache_key, cache_dir)

    except Exception as e:
        print(f"Error computing statistics: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
