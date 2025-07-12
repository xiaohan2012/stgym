import numpy as np
import pytest
import torch

from stgym.utils import (
    batch2ptr,
    collapse_ptr_list,
    flatten_dict,
    hsplit_and_vstack,
    mask_diagonal_sp,
    rand_ints,
    stacked_blocks_to_block_diagonal,
)

from .utils import DEVICE

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
    A = torch.tensor(A, device=DEVICE)
    ptr = torch.tensor(ptr, dtype=torch.int64, device=DEVICE)
    expected = torch.tensor(expected)
    actual = stacked_blocks_to_block_diagonal(A, ptr)

    assert actual.layout == torch.sparse_coo
    assert actual.device == A.device
    np.testing.assert_allclose(expected, actual.to_dense().cpu(), rtol=RTOL)


def test_mask_diagonal_sp():
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], device=DEVICE)
    values = torch.tensor([1, 2, 3, 4], device=DEVICE)
    A = torch.sparse_coo_tensor(indices, values, (2, 2), device=DEVICE).coalesce()
    masked_A = mask_diagonal_sp(A)
    assert masked_A.layout == torch.sparse_coo
    assert masked_A.device == A.device
    np.testing.assert_allclose(
        masked_A.to_dense().cpu(), torch.tensor([[0, 2], [3, 0]])
    )


def test_batch2ptr():
    batch = torch.tensor([0, 1, 1, 2, 2, 2], device=DEVICE)
    actual = batch2ptr(batch)
    assert actual.device == batch.device
    # actual = batch2ptr(torch.tensor([1, 2, 2, 3, 3, 3]))
    expected = torch.tensor([0, 1, 3, 6])
    np.testing.assert_allclose(actual.cpu(), expected)


def test_batch2ptr_with_error():
    with pytest.raises(
        ValueError, match=".*The batch contains zero-frequency element.*"
    ):
        batch2ptr(torch.tensor([1, 2, 2, 3, 3, 3], device=DEVICE))


def test_hsplit_and_vstack():
    A = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], device=DEVICE)
    chunk_size = 2
    actual = hsplit_and_vstack(A, chunk_size)
    assert actual.device == A.device
    expected = torch.tensor([[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]])
    np.testing.assert_allclose(actual.cpu(), expected)


@pytest.mark.parametrize(
    "input, output",
    [
        ({"a": {"b": "c"}}, {"a/b": "c"}),
        ({"a": {"b": "c"}, "d": "e"}, {"a/b": "c", "d": "e"}),
        ({"a": "bc"}, {"a": "bc"}),
        ({"a": [{"b": "c"}, {"b": "c"}]}, {"a/0/b": "c", "a/1/b": "c"}),
        ({"a": [{"b": "c"}, {"d": "e"}]}, {"a/0/b": "c", "a/1/d": "e"}),
        ({"a": [0, 1, 2]}, {"a": [0, 1, 2]}),
    ],
)
def test_flatten_dict(input, output):
    assert output == flatten_dict(input)


def test_random_ints():
    assert rand_ints(10).shape == (10,)
    np.testing.assert_allclose(rand_ints(10, seed=42), rand_ints(10, seed=42))
    assert not np.allclose(rand_ints(10, seed=42), rand_ints(10, seed=24))


@pytest.mark.parametrize(
    "ptr_list, expected",
    [
        (
            [
                torch.Tensor([0, 10, 20]),
                torch.Tensor([0, 20, 40]),
                torch.Tensor([0, 30, 60]),
            ],
            torch.Tensor([0, 10, 20, 40, 60, 90, 120]),
        ),
        ([torch.Tensor([0, 10, 20])], torch.Tensor([0, 10, 20])),
    ],
)
def test_collapse_ptr_list(ptr_list, expected):
    expected = expected.type(torch.int)
    actual = collapse_ptr_list(ptr_list)

    assert torch.allclose(expected, actual)
