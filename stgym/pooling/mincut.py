import torch
import torch.nn.functional as F
from torch import Tensor
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


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    batch: Tensor,
    s: Tensor,
    temp: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = adj.device
    print(device)
    if not is_sparse(adj):
        raise TypeError("adjacency matrix is not sparse")

    ptr = batch2ptr(batch)
    assert adj.ndim == 2
    assert batch.ndim == 1
    assert batch.shape[0] == adj.shape[0], f"{batch.shape[0]} != {adj.shape[0]}"
    assert ptr.ndim == 1
    assert s.ndim == 2

    n = ptr[1:] - ptr[:-1]
    # print(f"n: {n}")

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    K = s.shape[1]  # number of clusters
    B = ptr.shape[0] - 1  # number of blocks
    # print(f"B: {B}")
    # print(f"K: {K}")
    d = adj.sum(axis=0).to_dense()  # degree vector
    # print(f"d.shape: {d.shape}")
    range_k_times_b = torch.arange(K * B, device=device)
    diagonal_indices = torch.stack([range_k_times_b, range_k_times_b])

    # block diagonal matrices of C and d
    C_bd = stacked_blocks_to_block_diagonal(s, ptr, requires_grad=True)  # [N, K x B]
    print(f"C_bd.device: {C_bd.device}")
    range_n_sum = torch.arange(n.sum(), device=device)
    # print(f"C_bd.shape: {C_bd.shape}")
    d_diag = torch.sparse_coo_tensor(
        torch.stack([range_n_sum, range_n_sum]),
        d,
        requires_grad=True,
    )
    print(f"d_diag.device: {d_diag.device}")
    # print(f"d_diag.shape: {d_diag.shape}")

    # mincut loss
    mincut_normalizer = C_bd.T @ d_diag @ C_bd
    print(f"mincut_normalizer.device: {mincut_normalizer.device}")

    mincut_loss = -torch.trace(C_bd.T @ adj @ C_bd.to_dense()) / torch.trace(
        mincut_normalizer.to_dense()
    )
    print(f"mincut_loss.device: {mincut_loss.device}")

    sqrt_K = torch.sqrt(torch.tensor(K, device=device, dtype=torch.float))
    print(f"sqrt_K.device: {sqrt_K.device}")

    # orthogonality loss
    # CC = torch.sparse.mm(C_bd.T, C_bd)  # [KxB, KxB]
    # CC = C_bd.T @ C_bd  # [KxB, KxB]
    # Han: converting CC to dense to temporarly address RuntimeError: expand is unsupported for Sparse tensors
    CC = (C_bd.T @ C_bd).to_dense()  # [KxB, KxB]
    print(f"CC.device: {CC.device}")

    # normalization matrix
    CC_batch = torch.arange(0, B, device=device).repeat_interleave(K)
    CC_norm = torch.sqrt(
        scatter_sum(
            CC.pow(2).sum(axis=1).to_dense(),
            index=CC_batch,
        )
    )
    print(f"CC_norm.device: {CC_norm.device}")
    CC_normalizer = torch.sparse_coo_tensor(
        diagonal_indices,
        1 / CC_norm.repeat_interleave(K),
        requires_grad=True,
        device=device,
    )
    print(f"CC_normalizer.device(): {CC_normalizer.device()}")
    # construct the I_k matrix of shape [KxB, KxB], further divided by sqrt(K)
    I_div_k = torch.sparse_coo_tensor(
        diagonal_indices, torch.Tensor(1 / sqrt_K).repeat(K * B)
    )

    print(f"I_div_k.device: {I_div_k.device}")
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
    d_norm = torch.sparse_coo_tensor(
        diagonal_indices, (1 / d), requires_grad=False, device=device
    )
    out_adj_normalized = d_norm @ out_adj @ d_norm

    x_bd = stacked_blocks_to_block_diagonal(x, ptr)
    out_x = hsplit_and_vstack(s.T @ x_bd, chunk_size=x.shape[1])  # (BxK) x D
    return out_x, out_adj_normalized, mincut_loss, ortho_loss, CC_batch


class MincutPoolingLayer(torch.nn.Module):
    def __init__(self, cfg: PoolingConfig, **kwargs):
        super().__init__()
        # one linear layer mapping the input features to clustering space
        self.K = cfg.n_clusters
        self.linear = Linear(-1, self.K)

    def forward(self, batch):
        s = self.linear(batch.x)
        s = torch.softmax(s, dim=-1)
        # out_x = F.selu(s.T @ x_bd)  # S x (B x D)

        # out_x = hsplit_and_vstack(out_x, chunk_size=batch.x.shape[1])  # (SxB) x D

        (out_x, out_adj, mincut_loss, ortho_loss, out_batch) = mincut_pool(
            batch.x, batch.adj_t, batch.batch, s
        )

        batch.x = F.selu(out_x)  # apply selu activation function
        batch.adj_t = out_adj
        batch.batch = out_batch
        batch.ptr = batch2ptr(batch.batch)

        edge_index, edge_weight = to_undirected(*to_edge_index(out_adj), reduce="mean")
        batch.edge_index = edge_index
        batch.edge_weight = edge_weight
        # return batch, s, spectral_loss, cluster_loss, ortho_loss
        batch.s = s
        loss_info = {
            "mincut_loss": mincut_loss,
            "ortho_loss": ortho_loss,
        }
        attach_loss_to_batch(batch, loss_info)
        return batch
