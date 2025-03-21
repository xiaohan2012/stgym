import torch


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
