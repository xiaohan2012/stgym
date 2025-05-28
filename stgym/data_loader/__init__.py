import torch
import torch_geometric.transforms as T
from logzero import logger
from torch.utils.data import DataLoader, random_split
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.loader import DataLoader

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader.brca import BRCADataset
from stgym.data_loader.ds_info import get_info
from stgym.data_loader.human_crc import HumanCRCDataset


def load_dataset(task_cfg: TaskConfig, dl_cfg: DataLoaderConfig):
    """load dataset by name"""

    if dl_cfg.graph_const == "radius":
        radius = dl_cfg.radius_ratio * get_info(task_cfg.dataset_name)["max_span"]
        logger.debug("Using radius graph construction")
        logger.debug(f"Setting radius to {radius}")

    ds_name = task_cfg.dataset_name.lower()
    transform = T.compose.Compose(
        [
            (
                T.KNNGraph(k=dl_cfg.knn_k)
                if dl_cfg.graph_const == "knn"
                else T.RadiusGraph(r=radius)
            ),
            T.ToSparseTensor(
                remove_edge_index=False, layout=torch.sparse_coo
            ),  # keep edge_index
        ]
    )

    data_dir = f"./data/{ds_name}"
    if ds_name == "brca":
        return BRCADataset(root=data_dir, transform=transform)
    elif ds_name == "human-crc":
        ds = HumanCRCDataset(
            root="./data/human-crc",
            transform=transform,
        )
        return ds
    elif ds_name == "brca-test":
        ds = BRCADataset(
            root="./tests/data/brca-test",
            transform=transform,
            # keep only small graphs
            pre_filter=lambda g: g.num_nodes <= 500,
        )
        return ds
    elif ds_name == "human-crc-test":
        ds = HumanCRCDataset(
            root="./tests/data/human-crc-test",
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

    def __init__(self, task_cfg: TaskConfig, dl_cfg: DataLoaderConfig):
        self.ds = load_dataset(task_cfg, dl_cfg)
        self.loaders = create_loader(self.ds, dl_cfg)
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
