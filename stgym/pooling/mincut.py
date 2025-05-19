import torch
from torch import Tensor
from torch_geometric.utils.sparse import is_sparse
from torch_scatter import scatter_sum

from stgym.utils import (
    batch2ptr,
    hsplit_and_vstack,
    mask_diagonal_sp,
    stacked_blocks_to_block_diagonal,
)


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    batch: Tensor,
    s: Tensor,
    temp: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = adj.device

    if not is_sparse(adj):
        raise TypeError("adjacency matrix is not sparse")

    ptr = batch2ptr(batch)
    assert adj.ndim == 2
    assert batch.ndim == 1
    assert batch.shape[0] == adj.shape[0], f"{batch.shape[0]} != {adj.shape[0]}"
    assert ptr.ndim == 1
    assert s.ndim == 2

    n = ptr[1:] - ptr[:-1]
    print(f"n: {n}")

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    K = s.shape[1]  # number of clusters
    B = ptr.shape[0] - 1  # number of blocks
    print(f"B: {B}")
    print(f"K: {K}")
    d = adj.sum(axis=0).to_dense()  # degree vector
    print(f"d.shape: {d.shape}")
    diagonal_indices = torch.stack(
        [torch.arange(K * B).to(device), torch.arange(K * B).to(device)]
    )

    # block diagonal matrices of C and d
    C_bd = stacked_blocks_to_block_diagonal(s, ptr, requires_grad=True)  # [N, K x B]
    print(f"C_bd.shape: {C_bd.shape}")
    d_diag = torch.sparse_coo_tensor(
        torch.stack([torch.arange(n.sum()), torch.arange(n.sum())]).to(device),
        d,
        requires_grad=True,
    )
    print(f"d_diag.shape: {d_diag.shape}")

    # mincut loss
    mincut_normalizer = C_bd.T @ d_diag @ C_bd

    mincut_loss = -torch.trace(C_bd.T @ adj @ C_bd.to_dense()) / torch.trace(
        mincut_normalizer.to_dense()
    )

    sqrt_K = torch.sqrt(torch.tensor(K))

    # orthogonality loss
    # CC = torch.sparse.mm(C_bd.T, C_bd)  # [KxB, KxB]
    # CC = C_bd.T @ C_bd  # [KxB, KxB]
    # Han: converting CC to dense to temporarly address RuntimeError: expand is unsupported for Sparse tensors
    CC = (C_bd.T @ C_bd).to_dense()  # [KxB, KxB]

    # normalization matrix
    CC_batch = torch.arange(0, B).to(device).repeat_interleave(K)
    CC_norm = torch.sqrt(
        scatter_sum(
            CC.pow(2).sum(axis=1).to_dense(),
            index=CC_batch,
        )
    )
    CC_normalizer = torch.sparse_coo_tensor(
        diagonal_indices, 1 / CC_norm.repeat_interleave(K), requires_grad=True
    )
    # construct the I_k matrix of shape [KxB, KxB], further divided by sqrt(K)
    I_div_k = torch.sparse_coo_tensor(
        diagonal_indices, torch.Tensor(1 / sqrt_K).to(device).repeat(K * B)
    )

    # compute the norm of the block diagonal over graph batches
    ortho_loss = (
        scatter_sum(
            (CC @ CC_normalizer - I_div_k).pow(2).sum(axis=1).to_dense(), CC_batch
        )
        .sqrt()
        .mean()
    )

    # normalize the output adjacency matrix
    out_adj = C_bd.T @ adj @ C_bd
    out_adj = mask_diagonal_sp(out_adj)
    d = torch.einsum("ij->i", out_adj).to_dense().sqrt() + 1e-12
    d_norm = torch.sparse_coo_tensor(diagonal_indices, (1 / d), requires_grad=False)
    out_adj_normalized = d_norm @ out_adj @ d_norm

    x_bd = stacked_blocks_to_block_diagonal(x, ptr)
    out_x = hsplit_and_vstack(s.T @ x_bd, chunk_size=x.shape[1])  # (BxK) x D
    return out_x, out_adj_normalized, mincut_loss, ortho_loss, CC_batch
