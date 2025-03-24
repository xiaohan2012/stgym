import numpy as np
import pytest
import torch

from stgym.utils import (
    batch2ptr,
    hsplit_and_vstack,
    mask_diagonal_sp,
    stacked_blocks_to_block_diagonal,
)

RTOL = 1e-10


@pytest.mark.parametrize(
    "A, ptr, expected",
    [
        # 2 groups, 2 clusters
        (
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            [0, 2, 5],
            [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8], [0, 0, 9, 10]],
        ),
        # 3 groups, 2 clusters
        (
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            [0, 2, 4, 5],
            [
                [1, 2, 0, 0, 0, 0],
                [3, 4, 0, 0, 0, 0],
                [0, 0, 5, 6, 0, 0],
                [0, 0, 7, 8, 0, 0],
                [0, 0, 0, 0, 9, 10],
            ],
        ),
        # 2 groups, 1 cluster
        (
            [[1], [2], [3], [4], [5]],
            [0, 2, 5],
            [[1, 0], [2, 0], [0, 3], [0, 4], [0, 5]],
        ),
        # 3 groups, 1 cluster
        (
            [[1], [2], [3], [4], [5]],
            [0, 2, 4, 5],
            [[1, 0, 0], [2, 0, 0], [0, 3, 0], [0, 4, 0], [0, 0, 5]],
        ),
    ],
)
def test_stacked_blocks_to_block_diagonal(A, ptr, expected):
    A = torch.tensor(A)
    ptr = torch.tensor(ptr, dtype=torch.int64)
    expected = torch.tensor(expected)
    actual = stacked_blocks_to_block_diagonal(A, ptr)

    assert actual.layout == torch.sparse_coo
    np.testing.assert_allclose(expected, actual.to_dense(), rtol=RTOL)


def test_mask_diagonal_sp():
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    values = torch.tensor([1, 2, 3, 4])
    A = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
    masked_A = mask_diagonal_sp(A)
    assert masked_A.layout == torch.sparse_coo

    np.testing.assert_allclose(masked_A.to_dense(), torch.tensor([[0, 2], [3, 0]]))


def test_batch2ptr():
    actual = batch2ptr(torch.tensor([0, 1, 1, 2, 2, 2]))
    # actual = batch2ptr(torch.tensor([1, 2, 2, 3, 3, 3]))
    expected = torch.tensor([0, 1, 3, 6])
    np.testing.assert_allclose(actual, expected)


def test_batch2ptr_with_error():
    with pytest.raises(
        ValueError, match=".*The batch contains zero-frequency element.*"
    ):
        batch2ptr(torch.tensor([1, 2, 2, 3, 3, 3]))


def test_hsplit_and_vstack():
    A = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    chunk_size = 2
    actual = hsplit_and_vstack(A, chunk_size)
    expected = torch.tensor([[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]])
    np.testing.assert_allclose(actual, expected)
