import math
from os import path as osp

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense.mincut_pool import dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.sparse import is_sparse

from stgym.config_schema import PoolingConfig
from stgym.pooling.mincut import MincutPoolingLayer
from stgym.pooling.mincut import mincut_pool as sparse_mincut_pool
from stgym.utils import stacked_blocks_to_block_diagonal

from ..utils import BatchLoaderMixin

RTOL = 2e-3


def test_mincut_pool():
    path = osp.join("data", "TU")
    dataset = TUDataset(path, name="MUTAG").shuffle()

    B = 4
    train_loader = DataLoader(dataset[:0.9], B, shuffle=True)

    batch = next(iter(train_loader))
    n_nodes = batch.x.shape[0]
    n_features = batch.x.shape[1]
    print(f"n_features: {n_features}")
    s, t = batch.edge_index[0], batch.edge_index[1]
    adj = torch.sparse_coo_tensor(
        torch.stack([s, t]), torch.ones(s.size(0)), (n_nodes, n_nodes)
    )
    batch.adj_t = adj

    n_nodes = batch.x.shape[0]
    K = 3
    C = torch.rand(n_nodes, K)

    C_3d, mask = to_dense_batch(C, batch.batch)
    x_3d, mask = to_dense_batch(batch.x, batch.batch)
    adj_3d = to_dense_adj(batch.edge_index, batch=batch.batch)
    expected_out_x, expected_out_adj, expected_mincut_loss, expected_ortho_loss = (
        dense_mincut_pool(x_3d, adj_3d, C_3d, mask)
    )
    expected_batch = torch.arange(0, B).repeat_interleave(K)

    x_3d, mask = to_dense_batch(batch.x, batch.batch)

    (
        actual_out_x,
        actual_out_adj,
        actual_mincut_loss,
        actual_ortho_loss,
        actual_batch,
    ) = sparse_mincut_pool(batch.x, batch.adj_t, batch.batch, C)
    assert actual_out_x.shape == (B * K, n_features)

    np.testing.assert_allclose(
        actual_mincut_loss.detach().numpy(), expected_mincut_loss, rtol=RTOL
    )
    np.testing.assert_allclose(
        actual_ortho_loss.detach().numpy(), expected_ortho_loss, rtol=RTOL
    )

    np.testing.assert_allclose(
        actual_out_x, expected_out_x.reshape((-1, batch.x.shape[1]))
    )
    np.testing.assert_allclose(actual_batch, expected_batch)

    assert is_sparse(actual_out_adj)
    expected_out_adj_bd = stacked_blocks_to_block_diagonal(
        torch.vstack(list(expected_out_adj)), torch.arange(B + 1) * K
    ).to_dense()

    np.testing.assert_allclose(
        actual_out_adj.to_dense().detach().numpy(), expected_out_adj_bd, rtol=RTOL
    )


class TestAutoGrad(BatchLoaderMixin):
    def test(self):
        torch.autograd.set_detect_anomaly(True)
        num_clusters = 2

        batch = self.load_batch()
        print(f"batch.adj_t.device: {batch.adj_t.device}")
        print(f"batch.x.device: {batch.x.device}")
        print(f"batch.batch.device: {batch.batch.device}")
        n_nodes = batch.x.shape[0]
        C = torch.rand(n_nodes, num_clusters, requires_grad=True).to(batch.adj_t.device)
        print(f"C.device: {C.device}")
        (
            out_x,
            out_adj,
            mincut_loss,
            ortho_loss,
            output_batch,
        ) = sparse_mincut_pool(batch.x, batch.adj_t, batch.batch, C)

        print(f"out_x.device: {out_x.device}")
        print(f"out_adj.device: {out_adj.device}")
        print(f"output_batch.device: {output_batch.device}")
        print(f"mincut_loss.device: {mincut_loss.device}")
        print(f"ortho_loss.device: {ortho_loss.device}")
        loss = mincut_loss + ortho_loss

        loss.backward()


class TestWrapper(BatchLoaderMixin):
    def test(self):
        num_clusters = 3
        batch = self.load_batch()

        cfg = PoolingConfig(
            type="mincut",
            n_clusters=num_clusters,
        )
        # (self.out_channels, self.in_channels): ([10], 128)
        model = MincutPoolingLayer(cfg).to(batch.adj_t.device)
        # output_batch, s, spectral_loss, cluster_loss, ortho_loss = model(batch)
        output_batch = model(batch)
        assert output_batch.x.shape == (
            num_clusters * self.batch_size,
            self.num_features,
        )
        assert output_batch.adj_t.shape == (
            num_clusters * self.batch_size,
            num_clusters * self.batch_size,
        )

        assert batch.edge_index.size() == (
            2,
            num_clusters * (num_clusters - 1) * self.batch_size,
        )

        assert batch.s.size() == (
            self.batch_size * self.num_nodes_per_graph,
            num_clusters,
        )
        # print("batch.s.sum(axis=1).shape: {}".format(batch.s.sum(axis=1).shape))
        # print("torch.ones(num_clusters * self.batch_size).shape: {}".format(torch.ones(num_clusters * self.batch_size).shape))
        # assert torch.isclose(batch.s.sum(axis=1), torch.ones(num_clusters * self.batch_size)).all()
        assert batch.batch.size() == (self.batch_size * num_clusters,)
        np.testing.assert_allclose(
            batch.ptr.cpu(), torch.arange(self.batch_size + 1) * num_clusters
        )

        assert -1 <= batch.loss[0]["mincut_loss"] <= 0
        assert 0 <= batch.loss[0]["ortho_loss"] <= math.sqrt(2)
