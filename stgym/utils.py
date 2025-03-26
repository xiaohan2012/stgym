import torch
from torch_geometric.data import Data


def stacked_blocks_to_block_diagonal(
    A: torch.Tensor, ptr: torch.Tensor
) -> torch.sparse.Tensor:
    """
    Convert vertically stacked matrix blocks to a block diagonal matrix.

    the sizes of the blocks are given by ptr[1:] - ptr[:-1]
    """
    assert A.ndim == 2, A.ndim
    assert ptr.ndim == 1, ptr.ndim

    assert ptr[-1] == A.shape[0], f"{ptr[-1]} != {A.shape[0]}"
    b = ptr.shape[0] - 1
    n, k = A.shape

    ind0 = torch.arange(n).repeat(k, 1).T
    ind1 = torch.arange(k).repeat(n, 1)

    sizes = ptr[1:] - ptr[:-1]
    ind0_offset = torch.arange(start=0, end=k * b, step=k)
    ind0_offset_expanded = (
        ind0_offset.repeat_interleave(repeats=sizes, dim=0).repeat(k, 1).T
    )

    ind1 += ind0_offset_expanded

    indices = torch.vstack([ind0.flatten(), ind1.flatten()])
    values = A.flatten()
    return torch.sparse_coo_tensor(indices, values)


def mask_diagonal_sp(A: torch.sparse.Tensor) -> torch.sparse.Tensor:
    indices = A.indices()
    values = A.values()
    mask = indices[0] != indices[1]
    return torch.sparse_coo_tensor(indices[:, mask], values[mask], A.size())


def batch2ptr(batch: torch.Tensor) -> torch.Tensor:
    freq = torch.bincount(batch)
    if (freq == 0).any():
        raise ValueError(
            "The batch contains zero-frequency element, consider making this function more robust (refer to the unit tests)"
        )
    return torch.concat([torch.tensor([0]), freq.cumsum(dim=0)])


def hsplit_and_vstack(A: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """horizontally split A into chunks of size `chunk_size` and then vertically stack them"""
    return torch.vstack(torch.split(A, chunk_size, dim=1))


def get_edge_weight(batch: Data) -> torch.Tensor:
    return getattr(batch, "edge_weight")


def attach_loss_to_batch(batch: Data, loss_dict: dict[str, torch.Tensor]) -> Data:
    if hasattr(batch, "loss"):
        batch.loss.append(loss_dict)
    else:
        batch.loss = [loss_dict]
