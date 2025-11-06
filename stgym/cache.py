"""Cache utilities for STGym dataset statistics.

This module provides utilities for saving and loading dataset statistics
to/from cache files to speed up memory estimation.
"""

import json
from pathlib import Path
from typing import Optional

from stgym.mem_utils import DatasetStatistics


def get_cache_file_path(cache_key: str, cache_dir: Path) -> Path:
    """Get the cache file path for a given cache key."""
    return cache_dir / f"{cache_key}.json"


def generate_cache_key(
    dataset_name: str, graph_const: str, knn_k: int = None, radius_ratio: float = None
) -> str:
    """Generate a unique cache key for dataset configuration."""
    if graph_const == "knn":
        return f"{dataset_name}_knn_k{knn_k}"
    elif graph_const == "radius":
        return f"{dataset_name}_radius_r{radius_ratio}"
    else:
        raise ValueError(f"Unsupported graph construction method: {graph_const}")


def load_cached_statistics(
    cache_key: str, cache_dir: Path = None, raise_on_error: bool = False
) -> Optional[DatasetStatistics]:
    """Load dataset statistics from cache.

    Args:
        cache_key: Cache key for the dataset configuration
        cache_dir: Cache directory path (defaults to ./data/dataset_stats_cache)
        raise_on_error: If True, raise FileNotFoundError on missing/corrupted cache

    Returns:
        DatasetStatistics if cache exists and is valid, None otherwise

    Raises:
        FileNotFoundError: If raise_on_error=True and cache is missing or corrupted
    """
    if cache_dir is None:
        # Use absolute path relative to this file's directory
        cache_dir = Path(__file__).parent.parent / "data" / "dataset_stats_cache"

    cache_file = get_cache_file_path(cache_key, cache_dir)

    if not cache_file.exists():
        if raise_on_error:
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        return None

    try:
        with open(cache_file) as f:
            stats_dict = json.load(f)

        # Use Pydantic validation to parse and validate the data
        return DatasetStatistics.model_validate(stats_dict)
    except (json.JSONDecodeError, FileNotFoundError, ValueError):
        # If cache is corrupted, incomplete, or validation fails
        if raise_on_error:
            raise FileNotFoundError(f"Cache file not found or corrupted: {cache_file}")
        return None


def save_statistics_to_cache(
    stats: DatasetStatistics, cache_key: str, cache_dir: Path
) -> None:
    """Save dataset statistics to cache file."""
    cache_file = get_cache_file_path(cache_key, cache_dir)

    # Convert Pydantic model to dictionary
    stats_dict = stats.model_dump()

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(cache_file, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"✅ Statistics saved to: {cache_file}")


def load_statistics_from_cache(cache_key: str, cache_dir: Path) -> DatasetStatistics:
    """Load dataset statistics from cache file, raising error if not found."""
    return load_cached_statistics(cache_key, cache_dir, raise_on_error=True)
