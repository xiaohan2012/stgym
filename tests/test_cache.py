"""Tests for stgym.cache module."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

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

            expected_data = asdict(self.mock_stats)
            assert saved_data == expected_data


@patch("stgym.cache.load_cached_statistics")
class TestLoadStatisticsFromCache:
    """Test load_statistics_from_cache function."""

    def test_raises_error_when_file_not_exists(
        self, mock_load_cached: MagicMock
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = "nonexistent_dataset"
            cache_file = cache_dir / f"{cache_key}.json"

            with pytest.raises(
                FileNotFoundError, match=f"Cache file not found: {cache_file}"
            ):
                load_statistics_from_cache(cache_key, cache_dir)

            # Should not call load_cached_statistics since file doesn't exist
            mock_load_cached.assert_not_called()

    def test_raises_error_when_cache_corrupted(
        self, mock_load_cached: MagicMock
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = "corrupted_dataset"
            cache_file = cache_dir / f"{cache_key}.json"

            # Create a dummy cache file
            cache_file.touch()

            # Mock load_cached_statistics to return None (indicating corruption)
            mock_load_cached.return_value = None

            with pytest.raises(
                FileNotFoundError,
                match=f"Cache file not found or corrupted: {cache_file}",
            ):
                load_statistics_from_cache(cache_key, cache_dir)

            mock_load_cached.assert_called_once_with(cache_key, cache_dir)
