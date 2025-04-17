import torch_geometric.datasets as pg_datasets
import torch
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data.lightning.datamodule import LightningDataModule

from stgym.config_schema import DataLoaderConfig
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def load_dataset(cfg: DataLoaderConfig):
    """load dataset by name"""

    ds_name = cfg.dataset_name.lower()
    transform = T.compose.Compose(
        [
            T.KNNGraph(k=32),
            T.ToSparseTensor(
                remove_edge_index=False, layout=torch.sparse_coo
            ),  # keep edge_index
        ]
    )

    data_dir = f"./data/{ds_name}"
    if ds_name == "brca":
        from stgym.data_loader.brca import BRCADataset

        return BRCADataset(root=data_dir, transform=transform)
    elif ds_name == "brca-test":
        from stgym.data_loader.brca import BRCADataset
        ds = BRCADataset(
            root='./tests/data/brca-test',
            transform=transform,
            # keep only small graphs
            pre_filter=lambda g: g.num_nodes <= 500,
        )
        return ds
    else:
        raise NotImplementedError(ds_name)


def create_loader(
    ds, cfg: DataLoaderConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """create 3 data loaders corresponding to train/val/test split"""
    train_ds, val_ds, test_ds = random_split(
        ds, lengths=[cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio]
    )
    print("len(ds): {}".format(len(ds)))
    print("len(train_ds): {}".format(len(train_ds)))
    print("len(val_ds): {}".format(len(val_ds)))
    print("len(test_ds): {}".format(len(test_ds)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    # val/test data is in one batch
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class STDataModule(LightningDataModule):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading routines in GraphGym.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """

    def __init__(self, cfg: DataLoaderConfig):
        self.ds = load_dataset(cfg)
        self.loaders = create_loader(self.ds, cfg)
        super().__init__(has_val=True, has_test=True)

    @property
    def num_features(self):
        return self.ds.data.x.shape[1]

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]
