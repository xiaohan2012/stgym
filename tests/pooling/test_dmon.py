import math
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.sparse import is_sparse

from stgym.config_schema import PoolingConfig
from stgym.pooling.dmon import DMoNPoolingLayer, dmon_pool
from stgym.utils import stacked_blocks_to_block_diagonal

from ..utils import BatchLoaderMixin
from .dense_dmon import dense_dmon_pool

RTOL = 1e-4


def test_dmon_pool():
    path = osp.join("data", "TU")
    dataset = TUDataset(path, name="MUTAG").shuffle()

    B = 4
    train_loader = DataLoader(dataset[:0.9], B, shuffle=True)

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
    expected_batch = torch.arange(0, B).repeat_interleave(n_clusters)

    x_3d, mask = to_dense_batch(batch.x, batch.batch)

    F.selu(torch.matmul(s.transpose(1, 2), x_3d))

    (
        actual_out_adj,
        actual_spectral_loss,
        actual_cluster_loss,
        actual_ortho_loss,
        actual_batch,
    ) = dmon_pool(batch.adj, batch.batch, C)
    np.testing.assert_allclose(actual_spectral_loss, expected_spectral_loss, rtol=RTOL)
    np.testing.assert_allclose(actual_ortho_loss, expected_ortho_loss, rtol=RTOL)
    np.testing.assert_allclose(actual_batch, expected_batch)
    # np.testing.assert_allclose(actual_cluster_loss, expected_cluster_loss, rtol=RTOL)

    assert is_sparse(actual_out_adj)
    expected_out_adj_bd = stacked_blocks_to_block_diagonal(
        torch.vstack(list(out_adj)), torch.arange(B + 1) * n_clusters
    ).to_dense()

    np.testing.assert_allclose(
        actual_out_adj.to_dense(), expected_out_adj_bd, rtol=RTOL
    )


class TestWrapper(BatchLoaderMixin):
    def test(self):
        num_clusters = 3
        batch = self.load_batch()

        cfg = PoolingConfig(
            type="dmon",
            n_clusters=num_clusters,
        )
        # (self.out_channels, self.in_channels): ([10], 128)
        model = DMoNPoolingLayer(cfg)
        # output_batch, s, spectral_loss, cluster_loss, ortho_loss = model(batch)
        output_batch = model(batch)
        assert output_batch.x.shape == (
            num_clusters * self.batch_size,
            self.num_features,
        )
        assert output_batch.adj.shape == (
            num_clusters * self.batch_size,
            num_clusters * self.batch_size,
        )

        assert batch.edge_index.size() == (
            2,
            num_clusters * (num_clusters - 1) * self.batch_size,
        )

        assert batch.s.size() == (self.batch_size * self.num_nodes, num_clusters)
        assert batch.batch.size() == (self.batch_size * num_clusters,)
        np.testing.assert_allclose(
            batch.ptr, torch.arange(self.batch_size + 1) * num_clusters
        )

        assert -1 <= batch.loss[0]["spectral_loss"] <= 0.5
        assert 0 <= batch.loss[0]["ortho_loss"] <= math.sqrt(2)
        assert 0 <= batch.loss[0]["cluster_loss"] <= math.sqrt(num_clusters) - 1
