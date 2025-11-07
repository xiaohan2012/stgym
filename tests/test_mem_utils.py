"""Tests for stgym.mem_utils module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from stgym.cache import DatasetStatistics
from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.mem_utils import compute_dataset_statistics_using_config


class TestComputeDatasetStatisticsUsingConfig:
    """Test compute_dataset_statistics_using_config function."""

    @property
    def mock_task_cfg(self) -> TaskConfig:
        return TaskConfig(
            dataset_name="test_dataset",
            type="graph-classification",
            num_classes=2,
        )

    @property
    def mock_dl_cfg(self) -> DataLoaderConfig:
        return DataLoaderConfig(
            batch_size=32,
            device="cpu",
            graph_const="knn",
            knn_k=10,
            num_workers=0,
            split=DataLoaderConfig.DataSplitConfig(
                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
            ),
        )

    def create_mock_graph(
        self, num_nodes: int, num_edges: int, num_features: int
    ) -> Data:
        """Create a mock graph with specified dimensions."""
        # Create node features
        x = torch.randn(num_nodes, num_features)

        # Create edge indices - ensure we have exactly num_edges
        if num_edges > 0:
            # Create random edge indices
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
        else:
            # Empty edge index for graphs with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    @patch("stgym.mem_utils.load_dataset")
    def test_calculation_correctness(self, mock_load_dataset: MagicMock) -> None:
        # Create mock dataset with known statistics
        mock_graphs = [
            self.create_mock_graph(num_nodes=10, num_edges=20, num_features=5),
            self.create_mock_graph(num_nodes=15, num_edges=30, num_features=5),
            self.create_mock_graph(num_nodes=12, num_edges=25, num_features=5),
        ]
        mock_load_dataset.return_value = mock_graphs

        result = compute_dataset_statistics_using_config(
            self.mock_task_cfg, self.mock_dl_cfg
        )

        # Verify statistics are computed correctly
        assert isinstance(result, DatasetStatistics)
        assert result.num_features == 5
        assert result.num_graphs == 3
        assert result.avg_nodes == pytest.approx((10 + 15 + 12) / 3)  # 12.33
        assert result.avg_edges == pytest.approx((20 + 30 + 25) / 3)  # 25.0
        assert result.max_nodes == 15
        assert result.max_edges == 30

        # Verify load_dataset was called with correct arguments
        mock_load_dataset.assert_called_once_with(self.mock_task_cfg, self.mock_dl_cfg)

    @patch("stgym.mem_utils.load_dataset")
    def test_handles_graphs_without_edge_index(
        self, mock_load_dataset: MagicMock
    ) -> None:
        # Test with graphs that don't have edge_index attribute
        mock_graph = Data(x=torch.randn(10, 5))  # No edge_index
        mock_graphs = [mock_graph]
        mock_load_dataset.return_value = mock_graphs

        result = compute_dataset_statistics_using_config(
            self.mock_task_cfg, self.mock_dl_cfg
        )

        assert result.num_features == 5
        assert result.num_graphs == 1
        assert result.avg_nodes == pytest.approx(10.0)
        assert result.avg_edges == pytest.approx(0.0)  # No edges when no edge_index
        assert result.max_nodes == 10
        assert result.max_edges == 0

    @pytest.mark.parametrize(
        "graph_specs,expected_avg_nodes,expected_avg_edges",
        [
            # Test different graph sizes
            ([(5, 10), (10, 20), (15, 30)], 10.0, 20.0),
            # Test graphs with no edges (merged from separate test)
            ([(5, 0), (8, 10)], 6.5, 5.0),
            ([(1, 0), (2, 2), (3, 6)], 2.0, 2.67),
            ([(100, 500), (200, 1000)], 150.0, 750.0),
        ],
    )
    @patch("stgym.mem_utils.load_dataset")
    def test_various_graph_configurations(
        self,
        mock_load_dataset: MagicMock,
        graph_specs: list,
        expected_avg_nodes: float,
        expected_avg_edges: float,
    ) -> None:
        # Create mock graphs based on specifications
        mock_graphs = [
            self.create_mock_graph(num_nodes=nodes, num_edges=edges, num_features=4)
            for nodes, edges in graph_specs
        ]
        mock_load_dataset.return_value = mock_graphs

        result = compute_dataset_statistics_using_config(
            self.mock_task_cfg, self.mock_dl_cfg
        )

        assert result.avg_nodes == pytest.approx(expected_avg_nodes)
        assert result.avg_edges == pytest.approx(expected_avg_edges, rel=1e-2)
        assert result.num_features == 4
        assert result.num_graphs == len(graph_specs)
