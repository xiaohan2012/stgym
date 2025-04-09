import torch_geometric.datasets as pg_datasets
from torch.utils.data import DataLoader
from torch_geometric.data.lightning.datamodule import LightningDataModule

from stgym.config_schema import DataLoaderConfig
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader


def load_dataset(cfg: DataLoaderConfig):
    """load dataset by name"""

    ds_name = cfg.dataset_name.lower()
    if ds_name == "ba2motif":
        ds = pg_datasets.BA2MotifDataset(root="./data")
        print("ds.data: {}".format(ds.data))
        return pg_datasets.BA2MotifDataset(root="./data")


def create_loader(
    ds, cfg: DataLoaderConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """create 3 data loaders corresponding to train/val/test split"""
    train_ds, val_ds, test_ds = random_split(
        ds, lengths=[cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
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
    def dim_in(self):
        return self.ds.data.x.shape[1]

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]
