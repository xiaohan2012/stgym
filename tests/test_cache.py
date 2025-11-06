"""Tests for stgym.cache module."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from stgym.cache import load_statistics_from_cache, save_statistics_to_cache
from stgym.mem_utils import DatasetStatistics


class TestSaveStatisticsToCache:
    """Test save_statistics_to_cache function."""

    @property
    def mock_stats(self) -> DatasetStatistics:
        return DatasetStatistics(
            num_features=30,
            avg_nodes=160.7,
            avg_edges=1604.5,
            max_nodes=1228,
            max_edges=12280,
            num_graphs=4569,
        )

    def test_creates_file_and_saves_content_correctly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = "test_dataset_knn_k10"

            save_statistics_to_cache(self.mock_stats, cache_key, cache_dir)

            # Verify file was created
            cache_file = cache_dir / f"{cache_key}.json"
            assert cache_file.exists()

            # Verify file contents match DatasetStatistics
            with open(cache_file) as f:
                saved_data = json.load(f)

            expected_data = self.mock_stats.model_dump()
            assert saved_data == expected_data


class TestLoadStatisticsFromCache:
    """Test load_statistics_from_cache function."""

    def test_raises_error_when_file_not_exists(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = "nonexistent_dataset"
            cache_file = cache_dir / f"{cache_key}.json"

            with pytest.raises(
                FileNotFoundError, match=f"Cache file not found: {cache_file}"
            ):
                load_statistics_from_cache(cache_key, cache_dir)

    def test_raises_error_when_cache_corrupted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = "corrupted_dataset"
            cache_file = cache_dir / f"{cache_key}.json"

            # Create a corrupted cache file with invalid JSON
            with open(cache_file, "w") as f:
                f.write("invalid json content")

            with pytest.raises(
                FileNotFoundError,
                match=f"Cache file not found or corrupted: {cache_file}",
            ):
                load_statistics_from_cache(cache_key, cache_dir)
