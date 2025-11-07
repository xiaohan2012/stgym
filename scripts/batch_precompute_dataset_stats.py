#!/usr/bin/env python3
"""Batch precompute dataset statistics from design space configuration.

This script takes a Hydra design space configuration file and generates
dataset statistics cache files for all combinations of dataset_name and
data_loader graph construction parameters.

Usage:
    python scripts/batch_precompute_dataset_stats.py --config conf/design_space/node_clf.yaml
    python scripts/batch_precompute_dataset_stats.py --config conf/design_space/graph_clf.yaml --cache-dir ./custom_cache
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add the project root to sys.path so we can import stgym
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from stgym.cache import (
    generate_cache_key,
    get_cache_file_path,
    save_statistics_to_cache,
)
from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader.ds_info import get_info
from stgym.mem_utils import compute_dataset_statistics_using_config


def load_config(config_path: str) -> DictConfig:
    """Load Hydra configuration with proper inheritance handling."""
    config_path = Path(config_path).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    return cfg


def generate_combinations(config: DictConfig) -> List[Dict]:
    """Generate all combinations of dataset and graph construction parameters."""
    combinations = []

    # Extract parameters from config - they're already lists from the config
    datasets = config.task.dataset_name
    graph_consts = config.data_loader.graph_const
    knn_ks = config.data_loader.get("knn_k", [])
    radius_ratios = config.data_loader.get("radius_ratio", [])

    for dataset in datasets:
        for graph_const in graph_consts:
            if graph_const == "knn":
                # For KNN, iterate through all knn_k values
                for knn_k in knn_ks:
                    combinations.append(
                        {
                            "dataset_name": dataset,
                            "graph_const": graph_const,
                            "knn_k": knn_k,
                            "radius_ratio": None,
                        }
                    )
            elif graph_const == "radius":
                # For radius, iterate through all radius_ratio values
                for radius_ratio in radius_ratios:
                    combinations.append(
                        {
                            "dataset_name": dataset,
                            "graph_const": graph_const,
                            "knn_k": None,
                            "radius_ratio": radius_ratio,
                        }
                    )
    return combinations


def cache_exists(
    dataset: str, graph_const: str, knn_k: int, radius_ratio: float, cache_dir: Path
) -> bool:
    """Check if cache file already exists for the given combination."""
    cache_key = generate_cache_key(dataset, graph_const, knn_k, radius_ratio)
    cache_file = get_cache_file_path(cache_key, cache_dir)
    return cache_file.exists()


def generate_stats_for_combination(combination: Dict, cache_dir: Path) -> None:
    """Generate statistics for a single combination and save to cache."""
    dataset_name = combination["dataset_name"]
    graph_const = combination["graph_const"]
    knn_k = combination["knn_k"]
    radius_ratio = combination["radius_ratio"]

    print(f"Computing statistics for {dataset_name} with {graph_const} construction...")

    # Create task and dataloader configs
    ds_info = get_info(dataset_name)
    task_cfg = TaskConfig(
        dataset_name=dataset_name,
        type=ds_info["task_type"],
        num_classes=ds_info["num_classes"],
    )

    dl_cfg = DataLoaderConfig(
        device="cpu",
        graph_const=graph_const,
        knn_k=knn_k,
        radius_ratio=radius_ratio,
        num_workers=0,
    )

    # Compute statistics
    stats = compute_dataset_statistics_using_config(task_cfg, dl_cfg)

    # Save to cache
    cache_key = generate_cache_key(dataset_name, graph_const, knn_k, radius_ratio)
    save_statistics_to_cache(stats, cache_key, cache_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Batch precompute dataset statistics from design space configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to design space configuration file (e.g., conf/design_space/node_clf.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).parent.parent / "data" / "dataset_stats_cache"),
        help="Cache directory path",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recomputation even if cache exists"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Generate combinations
    combinations = generate_combinations(config)
    print(f"Generated {len(combinations)} combinations")

    cache_dir = Path(args.cache_dir)
    computed_count = 0
    skipped_count = 0

    # Process each combination
    for i, combination in enumerate(combinations, 1):
        dataset = combination["dataset_name"]
        graph_const = combination["graph_const"]
        knn_k = combination["knn_k"]
        radius_ratio = combination["radius_ratio"]

        # Create description string
        if graph_const == "knn":
            desc = f"{dataset} + {graph_const} + k={knn_k}"
        else:
            desc = f"{dataset} + {graph_const} + ratio={radius_ratio}"

        print(f"\nProcessing combination {i}/{len(combinations)}: {desc}")

        # Check if cache exists
        if (
            cache_exists(dataset, graph_const, knn_k, radius_ratio, cache_dir)
            and not args.force
        ):
            cache_key = generate_cache_key(dataset, graph_const, knn_k, radius_ratio)
            print(f"✅ Cache already exists, skipping: {cache_key}.json")
            skipped_count += 1
        else:
            try:
                generate_stats_for_combination(combination, cache_dir)
                computed_count += 1
            except Exception as e:
                print(f"❌ Error computing statistics: {e}")
                continue

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Summary: {len(combinations)} combinations processed")
    print(f"  Computed: {computed_count}")
    print(f"  Skipped: {skipped_count} (cache existed)")
    print(f"Cache directory: {cache_dir}")


if __name__ == "__main__":
    main()
