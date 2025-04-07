import torch_geometric.datasets as pg_datasets
from stgym.config_schema import DataLoaderConfig
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def load_dataset(cfg: DataLoaderConfig):
    ds_name = cfg.dataset_name.lower()
    if ds_name == "ba2motif":
        return pg_datasets.BA2MotifDataset(root="./data")


def create_loader(cfg: DataLoaderConfig):
    ds = load_dataset(cfg)
    train_ds, val_ds, test_ds = random_split(ds, lengths=[0.7, 0.15, 0.15])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
