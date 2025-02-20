import torch
from torch import Tensor


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Tensor | None = None,
    temp: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""MinCut pooling for sparse adjacency matrices.

    This class mirros its dense counter-part `dense_mincut_pool`

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Sparse adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        temp (float, optional): Temperature parameter for softmax function.
            (default: :obj:`1.0`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is None:
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)

    # we resort to a different way of doing matrix multiplication
    # on batches of 2D matrices
    # since using the following
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    # gives RuntimeError: expand is unsupported for Sparse tensors
    # TODO: consider faster approaches
    out_adj = torch.matmul(
        _bmm(s, adj),
        s,
    )
    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)

    # TODO: einsum on coo_matrix is slower than
    #       on csr_matrix (after some tweaks)
    #       consider using csr_matrix
    d_flat = torch.einsum("ijk->ij", adj)
    d = _sparse_rank3_diag(d_flat)

    mincut_den = _rank3_trace(
        torch.matmul(
            _bmm(s, d),
            s,
        )
    )

    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    # Cluster loss
    cluster_size = torch.einsum("ijk->ik", s)  # B x C
    cluster_loss = torch.norm(input=cluster_size, dim=1)
    cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
    cluster_loss = cluster_loss.mean()

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss, cluster_loss


def _bmm(t2: Tensor, t3: Tensor):
    """Batched matrix multiplication.
    Multiply a 2D dense tensor with a 3D sparse coo tensor.
    """
    dim1 = t3.shape[0]
    return torch.stack(
        [torch.matmul(t2.transpose(1, 2)[i], t3[i]) for i in range(dim1)],
        dim=0,
    )


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def _sparse_diag_2d(diag):
    n = diag.shape[0]
    return torch.sparse_coo_tensor([list(range(n)), list(range(n))], diag)


def _sparse_rank3_diag(d_flat):
    return torch.stack([_sparse_diag_2d(diag.to_dense()) for diag in d_flat], dim=0)
