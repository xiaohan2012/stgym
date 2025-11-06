"""Cache utilities for STGym dataset statistics.

This module provides utilities for saving and loading dataset statistics
to/from cache files to speed up memory estimation.
"""

import json
from dataclasses import asdict
from pathlib import Path

from stgym.mem_utils import DatasetStatistics, load_cached_statistics


def get_cache_file_path(cache_key: str, cache_dir: Path) -> Path:
    """Get the cache file path for a given cache key."""
    return cache_dir / f"{cache_key}.json"


def save_statistics_to_cache(
    stats: DatasetStatistics, cache_key: str, cache_dir: Path
) -> None:
    """Save dataset statistics to cache file."""
    cache_file = get_cache_file_path(cache_key, cache_dir)

    # Convert dataclass to dictionary using dataclasses.asdict
    stats_dict = asdict(stats)

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(cache_file, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"✅ Statistics saved to: {cache_file}")


def load_statistics_from_cache(cache_key: str, cache_dir: Path) -> DatasetStatistics:
    """Load dataset statistics from cache file."""
    cache_file = get_cache_file_path(cache_key, cache_dir)

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    # Use the function from mem_utils to load cached statistics
    cached_stats = load_cached_statistics(cache_key, cache_dir)
    if cached_stats is None:
        raise FileNotFoundError(f"Cache file not found or corrupted: {cache_file}")

    return cached_stats
