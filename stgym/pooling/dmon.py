import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_edge_index, to_undirected
from torch_geometric.utils.sparse import is_sparse
from torch_scatter import scatter_sum

from stgym.config_schema import PoolingConfig
from stgym.utils import (
    attach_loss_to_batch,
    batch2ptr,
    hsplit_and_vstack,
    mask_diagonal_sp,
    stacked_blocks_to_block_diagonal,
)

EPS = 1e-15


def dmon_pool(adj: torch.Tensor, batch: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    adj: N x N, adjacency matrix
    batch: N, the batch vector
    ptr: B+1  # not needed actually
    s: N x K, the clustering matrix
    """
    if not is_sparse(adj):
        raise TypeError("adjacency matrix is not sparse")

    ptr = batch2ptr(batch)
    assert adj.ndim == 2
    assert batch.ndim == 1
    assert batch.shape[0] == adj.shape[0], f"{batch.shape[0]} != {adj.shape[0]}"
    assert ptr.ndim == 1
    assert s.ndim == 2

    n = ptr[1:] - ptr[:-1]

    s = torch.softmax(s, dim=-1)

    K = s.shape[1]  # number of clusters
    B = ptr.shape[0] - 1  # number of blocks

    d = adj.sum(axis=0).to_dense()  # degree vector

    # compute the 1/2m term for each graph, laied out as a diagonal matrix
    # of shape BxK by BxK
    m2 = scatter_sum(d, batch)

    m_inv = torch.repeat_interleave(1 / m2, K)
    diagonal_indices = torch.stack([torch.arange(K * B), torch.arange(K * B)])
    m_inv_sp = torch.sparse_coo_tensor(diagonal_indices, m_inv)

    # block diagonal matrices of C and d
    C_bd = stacked_blocks_to_block_diagonal(s, ptr)  # [N, K x B]
    d_bd = stacked_blocks_to_block_diagonal(d.unsqueeze(0).T, ptr)

    # the normalizer
    # (d.T x).T (d.T x) / 2m
    # TODO: consider using torch_sparse.spspmm
    Ca = C_bd.T @ d_bd
    Cb = d_bd.T @ C_bd
    Cd2 = Ca @ Cb

    normalizer = m_inv_sp @ Cd2

    # C.T A C
    out_adj = C_bd.T @ adj @ C_bd

    # take the mean over the batch
    spectral_loss = -torch.trace((m_inv_sp @ (out_adj - normalizer)).to_dense()) / B

    # cluster loss (collapse regularization)
    sqrt_K = torch.sqrt(torch.tensor(K))

    cluster_size = C_bd.sum(axis=0).to_dense().reshape((B, K))  # B x K

    cluster_loss = (cluster_size.pow(2).sum(axis=1).sqrt() / n * sqrt_K - 1).mean()

    # orthogonality loss
    CC = C_bd.T @ C_bd  # [KxB, KxB]
    CC_batch = torch.arange(0, B).repeat_interleave(K)
    CC_norm = torch.sqrt(
        scatter_sum(
            CC.pow(2).sum(axis=1).to_dense(),
            index=CC_batch,
        )
    )

    # construct the I_k matrix of shape [KxB, KxB], further divided by sqrt(K)
    CC_normalizer = torch.sparse_coo_tensor(
        diagonal_indices, 1 / CC_norm.repeat_interleave(K)
    )
    I_div_k = torch.sparse_coo_tensor(
        diagonal_indices, torch.Tensor(1 / sqrt_K).repeat(K * B)
    )

    # compute the norm of the block diagonal over graph batches
    ortho_loss = (
        scatter_sum(
            (CC @ CC_normalizer - I_div_k).pow(2).sum(axis=1).to_dense(), CC_batch
        )
        .sqrt()
        .mean()
    )
    # normalize the out_adj
    out_adj = mask_diagonal_sp(out_adj)
    d = torch.einsum("ij->i", out_adj).to_dense().sqrt() + 1e-12
    d_norm = torch.sparse_coo_tensor(diagonal_indices, (1 / d))
    out_adj_normalized = d_norm @ out_adj @ d_norm
    return out_adj_normalized, spectral_loss, cluster_loss, ortho_loss, CC_batch


class DMoNPoolingLayer(torch.nn.Module):
    def __init__(self, cfg: PoolingConfig, **kwargs):
        super().__init__()
        # one linear layer mapping the input features to clustering space
        self.K = cfg.n_clusters
        self.linear = Linear(-1, self.K)

    def forward(self, batch):
        s = self.linear(batch.x)
        # s = torch.softmax(s, dim=-1)
        x_bd = stacked_blocks_to_block_diagonal(batch.x, batch.ptr)
        out_x = F.selu(s.T @ x_bd)  # S x (B x D)

        out_x = hsplit_and_vstack(out_x, chunk_size=batch.x.shape[1])  # (SxB) x D

        out_adj, spectral_loss, cluster_loss, ortho_loss, out_batch = dmon_pool(
            batch.adj, batch.batch, s
        )

        batch.x = out_x
        batch.adj = out_adj
        batch.batch = out_batch
        batch.ptr = batch2ptr(batch.batch)

        edge_index, edge_weight = to_undirected(*to_edge_index(out_adj), reduce="mean")
        batch.edge_index = edge_index
        batch.edge_weight = edge_weight
        # return batch, s, spectral_loss, cluster_loss, ortho_loss
        batch.s = s
        loss_info = {
            "spectral_loss": spectral_loss,
            "cluster_loss": cluster_loss,
            "ortho_loss": ortho_loss,
        }
        attach_loss_to_batch(batch, loss_info)
        return batch
