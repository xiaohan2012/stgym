import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pydash as _
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from stgym.config_schema import (
    ClusteringModelConfig,
    DataLoaderConfig,
    GraphClassifierModelConfig,
    NodeClassifierModelConfig,
    TaskConfig,
    TrainConfig,
)
from stgym.loss import compute_classification_loss
from stgym.model import STClusteringModel, STGraphClassifier, STNodeClassifier
from stgym.optimizer import create_optimizer_from_cfg, create_scheduler
from stgym.utils import collapse_ptr_list

# from torch_geometric.graphgym.loss import compute_loss
# from torch_geometric.graphgym.models.gnn import GNN
# from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
# from torch_geometric.graphgym.register import network_dict, register_network

# register_network('gnn', GNN)

torch.autograd.set_detect_anomaly(True)


@dataclass
class Split:
    train = "train"
    val = "validation"
    test = "test"


class STGymModule(pl.LightningModule):
    def __init__(
        self,
        dim_in,
        model_cfg: (
            GraphClassifierModelConfig
            | ClusteringModelConfig
            | NodeClassifierModelConfig
        ),
        train_cfg: TrainConfig,
        task_cfg: TaskConfig,
        dl_cfg: DataLoaderConfig,
        dim_out: int = None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.task_cfg = task_cfg
        if task_cfg.type == "graph-classification":
            assert dim_out is not None, "`dim_out` should be provided."
            self.model = STGraphClassifier(dim_in, dim_out, model_cfg)
        elif task_cfg.type == "node-clustering":
            self.model = STClusteringModel(dim_in, model_cfg)
        elif task_cfg.type == "node-classification":
            assert dim_out is not None, "`dim_out` should be provided."
            self.model = STNodeClassifier(dim_in, dim_out, model_cfg)
        else:
            raise ValueError(f"Unsupported task type: {task_cfg.type}")
        self.val_step_outputs = []
        self.test_step_outputs = []

        # has integer value only if kfold split is used
        self.kfold_split_index = (
            dl_cfg.split.split_index if dl_cfg.use_kfold_split else None
        )
        # self.save_hyperparameters()  # do not use this, as it records the pydantic models directly

    @property
    def use_kfold_split(self) -> bool:
        return self.kfold_split_index is not None

    def prefix_log_key(self, key: str) -> str:
        """Prefix log key by split info if needed."""
        if self.use_kfold_split:
            return f"split_{self.kfold_split_index}_{key}"
        return key

    def on_fit_start(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer_from_cfg(self.train_cfg.optim)(
            self.model.parameters()
        )
        # optimizer = create_optimizer(self.model.parameters(), self.train_cfg.optim)
        scheduler = create_scheduler(optimizer, self.train_cfg.lr_schedule)
        return [optimizer], [scheduler]

    def _shared_step(self, batch: Data, split: str) -> Dict:
        batch.split = split

        if self.task_cfg.type == "graph-classification":
            batch, pred_logits, layer_losses = self(batch)
            true = batch.y
            # the corner case of batch size = 1
            if pred_logits.ndim == 0:
                pred_logits = pred_logits.unsqueeze(-1)

            loss, pred_score = compute_classification_loss(pred_logits, true)

            # TODO: may attach different weights to the loss terms
            if len(layer_losses) > 0:
                pooling_loss = sum(layer_losses[-1].values())
            else:
                pooling_loss = 0
            step_end_time = time.time()

            return dict(
                loss=loss + pooling_loss,
                true=true,
                pred_score=pred_score.detach(),
                step_end_time=step_end_time,
            )
        elif self.task_cfg.type == "node-clustering":
            ptr = batch.ptr  # used to determine instance boundaries
            batch, pred, layer_losses = self(batch)
            clustering_related_loss = sum(layer_losses[-1].values())
            step_end_time = time.time()
            if split == Split.train:
                return dict(
                    loss=clustering_related_loss,
                    step_end_time=step_end_time,
                )
            else:
                true = batch.y
                return dict(
                    ptr=ptr.detach(),
                    loss=clustering_related_loss,
                    true=true,
                    pred_score=pred.detach(),
                    step_end_time=step_end_time,
                )
        elif self.task_cfg.type == "node-classification":
            batch, pred_logits, layer_losses = self(batch)
            true = batch.y
            loss, pred_score = compute_classification_loss(
                pred_logits, true.type(torch.long)
            )

            if len(layer_losses):
                loss += sum(layer_losses[-1].values())

            step_end_time = time.time()
            return dict(
                loss=loss,
                true=true,
                pred_score=pred_score.detach(),
                step_end_time=step_end_time,
            )
        else:
            raise NotImplementedError

    def training_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.train)
        self.log(
            self.prefix_log_key("train_loss"),
            output["loss"],
            prog_bar=True,
            batch_size=len(batch),
        )
        return output

    def validation_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.val)
        self.val_step_outputs.append(output)
        self.log(
            self.prefix_log_key("val_loss"),
            output["loss"],
            prog_bar=True,
            batch_size=len(batch),
        )
        return output

    def test_step(self, batch: Data, *args, **kwargs):
        output = self._shared_step(batch, split=Split.test)
        self.test_step_outputs.append(output)
        return output

    def _extract_pred_and_test_from_step_outputs(self, split: str):
        """extract the prediction and ground truth accumulated from validation steps"""
        if split == "val":
            outputs = self.val_step_outputs
        elif split == "test":
            outputs = self.test_step_outputs
        else:
            raise ValueError(split)

        true = torch.cat([output["true"] for output in outputs])
        pred = torch.cat([output["pred_score"] for output in outputs])
        if "ptr" not in outputs[0]:
            return true, pred
        else:
            ptr = collapse_ptr_list(_.map_(outputs, lambda x: x["ptr"]))
            return true, pred, ptr

    def _shared_epoch_end(self, split: Split):

        if self.task_cfg.type == "graph-classification":
            true, pred = self._extract_pred_and_test_from_step_outputs(split=split)
            if self.task_cfg.num_classes > 2:
                # multi-classs
                roc_auc = roc_auc_score(
                    true.cpu(),
                    F.softmax(pred, dim=1).numpy().cpu(),
                    multi_class="ovo",  # Han: ovr does not work for incomplete label values (typically in small batches and num_classes is not small)
                    labels=list(range(self.task_cfg.num_classes)),
                )
            else:
                roc_auc = roc_auc_score(true.cpu(), pred.cpu())
            self.log(self.prefix_log_key(f"{split}_roc_auc"), roc_auc, prog_bar=True)
        elif self.task_cfg.type == "node-clustering":
            true, pred, ptr_batch = self._extract_pred_and_test_from_step_outputs(
                split=split
            )
            true, pred, ptr_batch = true.cpu(), pred.cpu(), ptr_batch.cpu()
            nmi_scores = []
            ari_scores = []
            for start, end in zip(ptr_batch[:-1], ptr_batch[1:]):
                nmi_scores.append(
                    normalized_mutual_info_score(
                        true[start:end], pred[start:end, :].argmax(axis=1)
                    )
                )
                ari_scores.append(
                    adjusted_rand_score(
                        true[start:end], pred[start:end, :].argmax(axis=1)
                    )
                )
            self.log(
                self.prefix_log_key(f"{split}_nmi"), np.mean(nmi_scores), prog_bar=True
            )
            self.log(
                self.prefix_log_key(f"{split}_ari"), np.mean(ari_scores), prog_bar=True
            )
        elif self.task_cfg.type == "node-classification":
            true, pred = self._extract_pred_and_test_from_step_outputs(split=split)
            pred_argmax = pred.argmax(axis=1).cpu()
            true = true.cpu()
            acc = accuracy_score(true, pred_argmax)
            micro_f1_score = f1_score(true, pred_argmax, average="micro")
            self.log(self.prefix_log_key(f"{split}_accuracy"), acc, prog_bar=True)
            self.log(
                self.prefix_log_key(f"{split}_micro_f1_score"),
                micro_f1_score,
                prog_bar=True,
            )

        getattr(self, f"{split}_step_outputs").clear()

    def on_validation_epoch_end(self):
        return self._shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self._shared_epoch_end("test")

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)
