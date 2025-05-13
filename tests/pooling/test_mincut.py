from os import path as osp

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.dense.mincut_pool import dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.sparse import is_sparse

from stgym.pooling.mincut import mincut_pool as sparse_mincut_pool
from stgym.utils import stacked_blocks_to_block_diagonal

from ..utils import BatchLoaderMixin

RTOL = 1e-3


def test_mincut_pool():
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
    batch.adj_t = adj

    n_nodes = batch.x.shape[0]
    n_clusters = 3
    C = torch.rand(n_nodes, n_clusters)

    C_3d, mask = to_dense_batch(C, batch.batch)
    x_3d, mask = to_dense_batch(batch.x, batch.batch)
    adj_3d = to_dense_adj(batch.edge_index, batch=batch.batch)
    expected_out_x, expected_out_adj, expected_mincut_loss, expected_ortho_loss = (
        dense_mincut_pool(x_3d, adj_3d, C_3d, mask)
    )
    expected_batch = torch.arange(0, B).repeat_interleave(n_clusters)

    x_3d, mask = to_dense_batch(batch.x, batch.batch)

    (
        actual_out_x,
        actual_out_adj,
        actual_mincut_loss,
        actual_ortho_loss,
        actual_batch,
    ) = sparse_mincut_pool(batch.x, batch.adj_t, batch.batch, C)
    np.testing.assert_allclose(
        actual_mincut_loss.detach().numpy(), expected_mincut_loss, rtol=RTOL
    )
    np.testing.assert_allclose(
        actual_ortho_loss.detach().numpy(), expected_ortho_loss, rtol=RTOL
    )
    np.testing.assert_allclose(actual_out_x, expected_out_x)
    np.testing.assert_allclose(actual_batch, expected_batch)

    assert is_sparse(actual_out_adj)
    expected_out_adj_bd = stacked_blocks_to_block_diagonal(
        torch.vstack(list(expected_out_adj)), torch.arange(B + 1) * n_clusters
    ).to_dense()

    np.testing.assert_allclose(
        actual_out_adj.to_dense().detach().numpy(), expected_out_adj_bd, rtol=RTOL
    )


class TestAutoGrad(BatchLoaderMixin):
    def test(self):
        torch.autograd.set_detect_anomaly(True)
        num_clusters = 2

        batch = self.load_batch()

        n_nodes = batch.x.shape[0]
        C = torch.rand(n_nodes, num_clusters, requires_grad=True).to(batch.adj_t.device)

        (
            out_x,
            out_adj,
            mincut_loss,
            ortho_loss,
            output_batch,
        ) = sparse_mincut_pool(batch.adj_t, batch.batch, C)

        loss = mincut_loss + ortho_loss

        loss.backward()
