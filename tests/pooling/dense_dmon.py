from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.dense.mincut_pool import _rank3_trace

EPS = 1e-12


def dense_dmon_pool(
    s: Tensor,
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
    s = s.unsqueeze(0) if s.dim() == 2 else s
    # print(s.shape)
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

    # s = torch.softmax(s, dim=-1)

    (batch_size, num_nodes, C) = s.size()
    # print("batch_size, num_nodes, C", batch_size, num_nodes, C)
    if mask is None:
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=adj.device)

    mask = mask.view(batch_size, num_nodes, 1).to(s.dtype)
    s = s * mask

    # out = F.selu(torch.matmul(s.transpose(1, 2), x))
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Spectral loss:
    degrees = torch.einsum("ijk->ij", adj)  # B X N
    degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
    # print('degrees', degrees)
    degrees_t = degrees.transpose(1, 2)  # B x 1 x N

    m = torch.einsum("ijk->i", degrees) / 2  # B
    # print(m)
    m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

    ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
    cb = torch.matmul(degrees_t, s)  # B x 1 x C

    normalizer = torch.matmul(ca, cb) / 2 / m_expand
    decompose = out_adj - normalizer
    # print('sum.shape', out_adj.sum(axis=0).shape)
    # print(out_adj)
    # print(out_adj.sum(axis=0))
    # print('out_adj / 2 / m', out_adj / 2 / m)
    spectral_loss = -_rank3_trace(decompose) / 2 / m
    # print('decompose', decompose.shape)
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

    return s, out_adj, spectral_loss, ortho_loss, cluster_loss
