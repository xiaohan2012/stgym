import numpy as np
import torch
import torch_geometric.transforms as T
from logzero import logger
from sklearn.model_selection import KFold
from torch.utils.data import Subset, random_split
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.loader import DataLoader

from stgym.config_schema import DataLoaderConfig, TaskConfig
from stgym.data_loader.brca import BRCADataset
from stgym.data_loader.breast_cancer import BreastCancerDataset
from stgym.data_loader.ds_info import get_info  # noqa
from stgym.data_loader.human_crc import HumanCRCDataset
from stgym.data_loader.human_intestine import HumanIntestineDataset
from stgym.data_loader.human_lung import HumanLungDataset
from stgym.data_loader.mouse_kidney import MouseKidneyDataset
from stgym.data_loader.mouse_preoptic import MousePreopticDataset
from stgym.data_loader.mouse_spleen import MouseSpleenDataset


def get_dataset_class(ds_name: str):
    if ds_name in ("brca", "brca-test"):
        return BRCADataset
    elif ds_name == "breast-cancer":
        return BreastCancerDataset
    elif ds_name in ("human-crc", "human-crc-test"):
        return HumanCRCDataset
    elif ds_name == "mouse-spleen":
        return MouseSpleenDataset
    elif ds_name == "mouse-preoptic":
        return MousePreopticDataset
    elif ds_name == "mouse-kidney":
        return MouseKidneyDataset
    elif ds_name == "human-intestine":
        return HumanIntestineDataset
    elif ds_name == "human-lung":
        return HumanLungDataset
    else:
        raise NotImplementedError(f"{ds_name} is not available yet.")


def load_dataset(task_cfg: TaskConfig, dl_cfg: DataLoaderConfig):
    """load dataset by name"""

    if dl_cfg.graph_const == "radius":
        ds_info = get_info(task_cfg.dataset_name)
        radius = dl_cfg.radius_ratio * (ds_info["max_span"] - ds_info["min_span"])
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

    ds_cls = get_dataset_class(ds_name)
    if ds_name.endswith("-test"):
        print(f"./tests/data/{ds_name}")
        return ds_cls(
            root=f"./tests/data/{ds_name}",
            transform=transform,
            # keep only small graphs
            pre_filter=lambda g: g.num_nodes <= 500,
        )
    else:
        return ds_cls(root=f"./data/{ds_name}", transform=transform)


def create_loader(
    ds, cfg: DataLoaderConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """create 3 data loaders corresponding to train/val/test split"""
    train_ds, val_ds, test_ds = random_split(
        ds, lengths=[cfg.split.train_ratio, cfg.split.val_ratio, cfg.split.test_ratio]
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    # val/test data is in one batch
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    return train_loader, val_loader, test_loader


def create_kfold_loader(ds, cfg: DataLoaderConfig, k: int):
    """Create data loader with kfold split at fold k."""
    assert isinstance(
        cfg.split, DataLoaderConfig.KFoldSplitConfig
    ), f"Wrong split type {type(cfg.split)}"
    num_folds = cfg.split.num_folds
    assert 0 <= k < num_folds, f"{k} not in range: [0, {num_folds})"

    # Create KFold splitter
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Get train/val splits for fold k
    indices = np.arange(len(ds))
    fold_splits = list(kfold.split(indices))
    train_indices, val_indices = fold_splits[k]

    # Create datasets using Subset
    train_ds = Subset(ds, train_indices.tolist())
    val_ds = test_ds = Subset(
        ds, val_indices.tolist()
    )  # test data same as val for k-fold CV

    # Create DataLoaders with same parameters as create_loader
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    return train_loader, val_loader, test_loader


class STDataModuleBase(LightningDataModule):
    @property
    def num_features(self):
        assert hasattr(self, "ds")
        return self.ds.data.x.shape[1]

    def train_dataloader(self) -> DataLoader:
        assert hasattr(self, "loaders")
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        assert hasattr(self, "loaders")
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        assert hasattr(self, "loaders")
        return self.loaders[2]


class STDataModule(STDataModuleBase):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading routines.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """

    def __init__(self, task_cfg: TaskConfig, dl_cfg: DataLoaderConfig):
        self.ds = load_dataset(task_cfg, dl_cfg).to(dl_cfg.device)
        self.loaders = create_loader(self.ds, dl_cfg)
        super().__init__(has_val=True, has_test=True)


class STKfoldDataModule(LightningDataModule):
    """Data module supporing k-fold cross validation."""

    def __init__(self, task_cfg: TaskConfig, dl_cfg: DataLoaderConfig):
        self.ds = load_dataset(task_cfg, dl_cfg).to(dl_cfg.device)
        self.dl_cfg = dl_cfg
        super().__init__(has_val=True, has_test=True)

    def create_loader_at_fold(self, k: int):
        self.loaders = create_kfold_loader(self.ds, self.dl_cfg, k=k)

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
