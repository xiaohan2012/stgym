import torch_geometric.datasets as pg_datasets

from stgym.config_schema import DataLoaderConfig

from .loader import create_loader  # noqa


def load_dataset(cfg: DataLoaderConfig):
    ds_name = cfg.dataset_name.lower()
    if ds_name == "ba2motif":
        return pg_datasets.BA2MotifDataset(root="./data")
