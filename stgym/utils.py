import os
import shutil
from pathlib import Path

import mlflow
import numpy as np
import ray
import torch
import yaml
from logzero import logger
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def stacked_blocks_to_block_diagonal(
    A: torch.Tensor, ptr: torch.Tensor, requires_grad: bool = False
) -> torch.sparse.Tensor:
    """
    Convert vertically stacked matrix blocks to a block diagonal matrix.

    the sizes of the blocks are given by ptr[1:] - ptr[:-1]
    """
    assert A.ndim == 2, A.ndim
    assert ptr.ndim == 1, ptr.ndim
    device = A.device

    assert ptr[-1] == A.shape[0], f"{ptr[-1]} != {A.shape[0]}"
    b = ptr.shape[0] - 1
    n, k = A.shape

    ind0 = torch.arange(n).repeat(k, 1).T.to(device)
    ind1 = torch.arange(k).repeat(n, 1).to(device)

    sizes = ptr[1:] - ptr[:-1]
    ind0_offset = torch.arange(start=0, end=k * b, step=k).to(device)
    ind0_offset_expanded = (
        ind0_offset.repeat_interleave(repeats=sizes, dim=0).repeat(k, 1).T
    )

    ind1 += ind0_offset_expanded

    indices = torch.vstack([ind0.flatten(), ind1.flatten()])
    values = A.flatten()
    return torch.sparse_coo_tensor(indices, values, requires_grad=requires_grad)


def mask_diagonal_sp(A: torch.sparse.Tensor) -> torch.sparse.Tensor:
    indices = A.indices()
    values = A.values()
    mask = indices[0] != indices[1]
    return torch.sparse_coo_tensor(indices[:, mask], values[mask], A.size())


def batch2ptr(batch: torch.Tensor) -> torch.Tensor:
    device = batch.device
    freq = torch.bincount(batch)
    if (freq == 0).any():
        raise ValueError(
            "The batch contains zero-frequency element, consider making this function more robust (refer to the unit tests)"
        )
    return torch.concat([torch.tensor([0]).to(device), freq.cumsum(dim=0)])


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


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        data = yaml.safe_load(file)
    return data


def flatten_dict(nested_dict, separator="/"):
    flat_dict = {}

    def aux(data, parent_key):
        for k, v in data.items():
            key = parent_key + separator + str(k) if str(parent_key) else k
            if isinstance(v, dict):
                aux(v, key)
            elif isinstance(v, list) and isinstance(
                v[0], dict
            ):  # recurse only if all elements are dict
                padded_v = dict(enumerate(v))
                for idx in range(len(padded_v)):
                    padded_key = key + separator + str(idx)
                    aux(padded_v[idx], padded_key)
            else:
                flat_dict[key] = v

    aux(nested_dict, "")
    return flat_dict


class RayProgressBar:
    @staticmethod
    def num_jobs_done_iter(obj_ids):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    @staticmethod
    def show(obj_ids):
        seq = RayProgressBar.num_jobs_done_iter(obj_ids)
        for x in tqdm(seq, total=len(obj_ids)):
            pass

    @staticmethod
    def check():
        assert ray.is_initialized()


def rand_ints(size, min=0, max=100000, seed: int = None) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(min, max, size)


def rm_dir_if_exists(dirname: str | Path):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def collapse_ptr_list(ptr_list: list[torch.Tensor]):
    """given a list of pointer arrays, collapse them into one array and add offset to the elements accordingly"""

    offset = 0
    ptr_list_adjusted = []
    for ptr in ptr_list:
        ptr_list_adjusted += ptr[1:] + offset
        offset = ptr_list_adjusted[-1]
    ptr_list_adjusted.insert(0, 0)  # prepend 0
    return torch.Tensor(ptr_list_adjusted).type(torch.int)


def create_mlflow_experiment(exp_name: str):
    """create a MLFlow experiment"""
    try:
        mlflow.create_experiment(exp_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Experiment '{exp_name}' already exists with ID: {experiment_id}"
            )
        else:
            logger.error(
                "Could not find the experiment even after error. Please check your setup."
            )


class YamlLoaderMixin:

    @classmethod
    def from_yaml(cls, yaml_file):
        data = load_yaml(yaml_file)

        return cls.model_validate(data)


def get_maximum_coord_span(ds: InMemoryDataset) -> float:
    """get the maximum cooridnate span from datums in a dataset"""
    span_list = []
    for dt in ds:
        xy_min, _ = dt.pos.min(axis=0)
        xy_max, _ = dt.pos.max(axis=0)
        span_list.append((xy_max - xy_min).max().numpy())
    logger.debug(f"min span: {min(span_list)}")
    logger.debug(f"max span: {max(span_list)}")
    return float(max(span_list))
