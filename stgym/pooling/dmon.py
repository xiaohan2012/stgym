from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.dense.mincut_pool import _rank3_trace
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum

from stgym.config_schema import PoolingConfig
from stgym.utils import batch2ptr, mask_diagonal_sp, stacked_blocks_to_block_diagonal

EPS = 1e-15


def dmon_pool(adj: torch.Tensor, batch: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    adj: N x N, adjacency matrix
    batch: N, the batch vector
    ptr: B+1  # not needed actually
    s: N x K, the clustering matrix
    """
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
    cluster_loss = (torch.linalg.norm(cluster_size, axis=1) / n * K - 1).mean()

    # orthogonality loss
    CC = C_bd.T @ C_bd  # [KxB, KxB]
    CC_batch = torch.arange(0, B).repeat_interleave(K)
    CC_norm = torch.sqrt(
        scatter_sum(
            CC.pow(2).sum(axis=1).to_dense(),
            index=CC_batch,
        )
    )
    CC_norm.repeat_interleave(K)

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
    return out_adj_normalized, spectral_loss, cluster_loss, ortho_loss


class DMoNPooling(torch.nn.Module):
    def __init__(self, k: int):
        """
        k: the number of clusters
        """
        super().__init__()

        self.linear = Linear(-1, k)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.linear.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.linear(x)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        degrees = torch.einsum("ijk->ij", adj)  # B X N
        degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N

        m = torch.einsum("ijk->i", degrees) / 2  # B
        m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x C

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(C).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
            dim=(-1, -2),
        )
        ortho_loss = ortho_loss.mean()

        # Cluster loss:
        cluster_size = torch.einsum("ijk->ik", s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(C, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum("ijk->ij", out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.linear.in_channels}, "
            f"num_clusters={self.linear.out_channels})"
        )


class DMoNPoolingLayer(torch.nn.Module):
    def __init__(self, cfg: PoolingConfig, **kwargs):
        super().__init__()
        # one linear layer
        self.model = DMoNPooling(k=cfg.n_clusters)

    def forward(self, batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        # print("mask: {}".format(mask))
        adj, mask = to_dense_batch(batch.adj, batch.batch)
        # print("mask: {}".format(mask))
        # print("adj: {}".format(adj))
        (_, out_x, out_adj, spectral_loss, ortho_loss, cluster_loss) = self.model(
            x, adj, mask
        )

        batch.x = out_x
        batch.adj = out_adj
        return batch
