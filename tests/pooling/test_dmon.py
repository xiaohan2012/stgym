import math

import numpy as np
import torch

from stgym.config_schema import PoolingConfig
from stgym.pooling.dmon import DMoNPooling, DMoNPoolingLayer, dmon_pool

from ..utils import BatchLoaderMixin
from .dense_dmon import dense_dmon_pool

RTOL = 1e-4


def test_dmon_pool():
    from os import path as osp

    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_adj, to_dense_batch

    path = osp.join("data", "TU")
    dataset = TUDataset(path, name="MUTAG").shuffle()

    train_loader = DataLoader(dataset[:0.9], 4, shuffle=True)

    batch = next(iter(train_loader))
    n_nodes = batch.x.shape[0]
    s, t = batch.edge_index[0], batch.edge_index[1]
    adj = torch.sparse_coo_tensor(
        torch.stack([s, t]), torch.ones(s.size(0)), (n_nodes, n_nodes)
    )
    batch.adj = adj

    n_nodes = batch.x.shape[0]
    n_clusters = 3
    C = torch.rand(n_nodes, n_clusters)

    C_3d, mask = to_dense_batch(C, batch.batch)
    adj_3d = to_dense_adj(batch.edge_index, batch=batch.batch)
    s, out_adj, expected_spectral_loss, expected_ortho_loss, expected_cluster_loss = (
        dense_dmon_pool(C_3d, adj_3d, mask)
    )

    actual_spectral_loss, actual_cluster_loss, actual_ortho_loss = dmon_pool(
        batch.adj, batch.batch, batch.ptr, C
    )
    np.testing.assert_allclose(actual_spectral_loss, expected_spectral_loss, rtol=RTOL)
    np.testing.assert_allclose(actual_ortho_loss, expected_ortho_loss, rtol=RTOL)
    # np.testing.assert_allclose(actual_cluster_loss, expected_cluster_loss, rtol=RTOL)


class TestDmonPooling:
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)

    def create_data(self):
        x = torch.randn((self.batch_size, self.num_nodes, self.channels))
        adj = torch.ones((self.batch_size, self.num_nodes, self.num_nodes))
        mask = torch.randint(0, 2, (self.batch_size, self.num_nodes), dtype=torch.bool)
        return x, adj, mask

    def test(self):
        x, adj, mask = self.create_data()

        pool = DMoNPooling(self.num_clusters)
        assert str(pool) == "DMoNPooling(-1, num_clusters=10)"

        s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(x, adj, mask)
        assert s.size() == (self.batch_size, self.num_nodes, self.num_clusters)
        assert x.size() == (self.batch_size, self.num_clusters, self.channels)
        assert adj.size() == (self.batch_size, self.num_clusters, self.num_clusters)
        assert -1 <= spectral_loss <= 0.5
        assert 0 <= ortho_loss <= math.sqrt(2)
        assert 0 <= cluster_loss <= math.sqrt(self.num_clusters) - 1


class TestWrapper(BatchLoaderMixin):
    def test(self):
        num_clusters = 10
        batch = self.load_batch()
        cfg = PoolingConfig(
            type="dmon",
            n_clusters=num_clusters,
        )
        # (self.out_channels, self.in_channels): ([10], 128)
        model = DMoNPoolingLayer(cfg)
        output_batch = model(batch)
        assert output_batch.x.shape == (
            self.batch_size,
            num_clusters,
            self.num_features,
        )
        assert output_batch.adj.shape == (self.batch_size, num_clusters, num_clusters)
